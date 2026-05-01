"""
Phase 4 — Train the concept-gated Chain-of-Thought model on cached BERT features.

Key differences vs. Phase 3:
  * Uses CoTModel (3 concept-gated attention heads + aux losses)
  * Loads subword-aligned concept masks (Phase 4 concept_masks.py output)
  * Multi-task loss:
        L = CE(main)  +  λ_aux * Σ CE(aux_head)  +  λ_cov * coverage penalty
    where coverage penalty encourages each head's mean score over examples
    that *contain* the concept to rise above 0.3 (dead-head prevention).
  * Saves per-epoch mean scores and rationale samples for inspection.

Usage:
    python src/phase4/train_phase4.py [--L 128] [--no-meta] [--seed 42]

Preconditions (run in order):
    python src/phase1/step6_concept_mapping.py
    python src/phase1/step7_dataset_split.py
    python src/phase2/step3_embeddings.py
    python src/phase4/concept_masks.py 128
"""
from __future__ import annotations

import os
import sys
import csv
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, accuracy_score, classification_report

_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _src)
from config import (
    DATA_DIR, MODELS_DIR, LOGS_DIR, RESULTS_DIR, BERT_MAX_LEN,
    TRAIN_BATCH_SIZE, TRAIN_LR, TRAIN_WEIGHT_DECAY, TRAIN_DROPOUT,
    TRAIN_NUM_EPOCHS, EARLY_STOP_PATIENCE, GRAD_CLIP_NORM, RANDOM_SEED,
    CONCEPT_AUX_LAMBDA, get_device,
)
from bert_features import load_features
from balanced_sampling import build_balanced_sampler, describe_buckets
from meta_encoder import (
    PARTY_VOCAB, CREDIT_COLS, META_TOTAL_DIM,
    fit_metadata, encode_metadata, describe_vocabs,
)
sys.path.insert(0, os.path.join(_src, "phase3"))
from phase4.cot_model import CoTModel
from phase4.concept_masks import load_subword_masks, cache_subword_masks
from phase4.rationale import generate_batch_rationales


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class CoTDataset(Dataset):
    def __init__(
        self,
        features_npz: str,
        masks_npz: str,
        meta: np.ndarray | None = None,
        has_meta: np.ndarray | None = None,
    ):
        f = load_features(features_npz)
        m = load_subword_masks(masks_npz)
        assert f["features"].shape[0] == m["emotion"].shape[0], (
            f"row mismatch: feat={f['features'].shape[0]} masks={m['emotion'].shape[0]}"
        )
        self.features = torch.from_numpy(f["features"])                         # float32
        self.attn     = torch.from_numpy(f["attention_mask"]).long()            # int
        self.labels   = torch.from_numpy(f["labels"]).long()
        self.em       = torch.from_numpy(m["emotion"])
        self.mo       = torch.from_numpy(m["modality"])
        self.ne       = torch.from_numpy(m["negation"])
        self.meta     = torch.from_numpy(meta).float() if meta is not None else None
        self.has_meta = (
            torch.from_numpy(has_meta).float() if has_meta is not None else None
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        base = (
            self.features[i], self.attn[i],
            self.em[i], self.mo[i], self.ne[i],
            self.labels[i],
        )
        if self.meta is not None:
            hm = self.has_meta[i] if self.has_meta is not None else torch.tensor(1.0)
            return base + (self.meta[i], hm)
        return base


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------
class LabelSmoothingCE(nn.Module):
    def __init__(self, smoothing: float = 0.05, weight: torch.Tensor | None = None):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight

    def forward(self, logits, targets):
        n_classes = logits.size(-1)
        with torch.no_grad():
            soft = torch.full_like(logits, self.smoothing / (n_classes - 1))
            soft.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        log_p = F.log_softmax(logits, dim=-1)
        loss = -(soft * log_p).sum(dim=-1)
        if self.weight is not None:
            loss = loss * self.weight[targets]
        return loss.mean()


def coverage_penalty(score: torch.Tensor, mask: torch.Tensor, target: float = 0.3) -> torch.Tensor:
    """
    Penalty that keeps each head 'awake' on examples that DO contain the concept.
    score: (B, 1)   mask: (B, T) — nonzero row means concept is present.
    If the mean score across present-examples < target, penalize linearly.
    """
    present = (mask.sum(dim=1) > 0).float()
    if present.sum() < 1:
        return torch.zeros((), device=score.device)
    mean_s = (score.squeeze(-1) * present).sum() / present.sum().clamp(min=1.0)
    return F.relu(target - mean_s)


def gated_aux_loss(aux_logits, targets, mask, weight=None):
    """
    Auxiliary CE that ONLY trains on examples where the concept is present.
    Without this, the emotion head's aux classifier (trained on all examples
    even when emotion-context is zero) degenerates to predicting the prior —
    which is what produced the 'dead emotion head' (E≈0.02) in earlier runs.
    """
    present = (mask.sum(dim=1) > 0)
    if present.sum() < 1:
        return torch.zeros((), device=aux_logits.device)
    return F.cross_entropy(aux_logits[present], targets[present], weight=weight)


# ---------------------------------------------------------------------------
# Step / evaluate
# ---------------------------------------------------------------------------
def _unpack(batch, use_meta, device):
    if use_meta:
        feats, amask, em, mo, ne, y, meta, has_meta = batch
        return (feats.to(device), amask.to(device),
                em.to(device), mo.to(device), ne.to(device),
                y.to(device), meta.to(device), has_meta.to(device))
    feats, amask, em, mo, ne, y = batch
    return (feats.to(device), amask.to(device),
            em.to(device), mo.to(device), ne.to(device),
            y.to(device), None, None)


@torch.no_grad()
def evaluate(model, loader, main_criterion, use_meta, device):
    model.eval()
    losses, preds, ys, hms = [], [], [], []
    em_avg, mo_avg, ne_avg = [], [], []
    for batch in loader:
        feats, amask, em_m, mo_m, ne_m, y, meta, has_meta = _unpack(batch, use_meta, device)
        out = model(feats, amask, em_m, mo_m, ne_m, meta, has_meta=has_meta)
        logits, em_s, mo_s, ne_s = out[0], out[1], out[2], out[3]
        losses.append(main_criterion(logits, y).item())
        preds.append(logits.argmax(dim=-1).cpu().numpy())
        ys.append(y.cpu().numpy())
        em_avg.append(em_s.mean().item())
        mo_avg.append(mo_s.mean().item())
        ne_avg.append(ne_s.mean().item())
        if has_meta is not None:
            hms.append(has_meta.cpu().numpy())
    preds = np.concatenate(preds); ys = np.concatenate(ys)
    has_meta_arr = np.concatenate(hms) if hms else None
    return {
        "loss":     float(np.mean(losses)),
        "acc":      float(accuracy_score(ys, preds)),
        "macro_f1": float(f1_score(ys, preds, average="macro")),
        "em_mean":  float(np.mean(em_avg)),
        "mo_mean":  float(np.mean(mo_avg)),
        "ne_mean":  float(np.mean(ne_avg)),
        "preds":    preds,
        "labels":   ys,
        "has_meta": has_meta_arr,
    }


def per_domain_breakdown(preds, ys, has_meta_arr):
    """LIAR (has_meta=1) vs CoAID (has_meta=0) acc / macro_f1."""
    if has_meta_arr is None:
        return None
    out = {}
    for name, mask_val in (("liar", 1.0), ("coaid", 0.0)):
        m = (has_meta_arr == mask_val)
        n = int(m.sum())
        if n < 1:
            continue
        out[name] = {
            "n":        n,
            "acc":      float(accuracy_score(ys[m], preds[m])),
            "macro_f1": float(f1_score(ys[m], preds[m], average="macro")),
        }
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _paths(L: int, dataset: str) -> dict:
    if dataset == "liar":
        feat_pre = "bert_features"
        mask_pre = "concept_masks"
        split_pre = ""
    else:
        feat_pre = f"bert_features_{dataset}"
        mask_pre = f"concept_masks_{dataset}"
        split_pre = f"{dataset}_"
    return {
        "feat_train": os.path.join(DATA_DIR, f"{feat_pre}_train_L{L}.npz"),
        "feat_val":   os.path.join(DATA_DIR, f"{feat_pre}_val_L{L}.npz"),
        "feat_test":  os.path.join(DATA_DIR, f"{feat_pre}_test_L{L}.npz"),
        "mask_train": os.path.join(DATA_DIR, f"{mask_pre}_train_L{L}.npz"),
        "mask_val":   os.path.join(DATA_DIR, f"{mask_pre}_val_L{L}.npz"),
        "mask_test":  os.path.join(DATA_DIR, f"{mask_pre}_test_L{L}.npz"),
        "csv_train":  os.path.join(DATA_DIR, f"{split_pre}train_split_L{L}.csv"),
        "csv_val":    os.path.join(DATA_DIR, f"{split_pre}val_split_L{L}.csv"),
        "csv_test":   os.path.join(DATA_DIR, f"{split_pre}test_split_L{L}.csv"),
    }


def main(
    L: int = 128,
    use_meta: bool = True,
    seed: int = RANDOM_SEED,
    dataset: str = "liar",
    balance: bool = False,
):
    torch.manual_seed(seed); np.random.seed(seed)
    device = get_device()
    print(f"\nPhase 4 training (dataset={dataset}, L={L}, meta={use_meta}, seed={seed}, device={device})")

    paths = _paths(L, dataset)

    # ----- Build concept-mask caches on demand -----
    for split, csv_p, feat_p, mask_p in [
        ("train", paths["csv_train"], paths["feat_train"], paths["mask_train"]),
        ("val",   paths["csv_val"],   paths["feat_val"],   paths["mask_val"]),
        ("test",  paths["csv_test"],  paths["feat_test"],  paths["mask_test"]),
    ]:
        cache_subword_masks(
            split_csv=csv_p,
            features_npz=feat_p,
            out_npz=mask_p,
            bert_max_len=BERT_MAX_LEN,
            overwrite=False,
        )

    # ----- Metadata -----
    meta_dim = 0
    meta_vocabs = None
    train_meta = val_meta = test_meta = None
    train_has = val_has = test_has = None
    if use_meta:
        train_csv = pd.read_csv(paths["csv_train"])
        val_csv   = pd.read_csv(paths["csv_val"])
        test_csv  = pd.read_csv(paths["csv_test"])
        # Fit on TRAIN ONLY, restricted to has_meta=1 rows internally
        meta_vocabs, scaler = fit_metadata(train_csv)
        train_meta = encode_metadata(train_csv, meta_vocabs, scaler)
        val_meta   = encode_metadata(val_csv,   meta_vocabs, scaler)
        test_meta  = encode_metadata(test_csv,  meta_vocabs, scaler)
        meta_dim = train_meta.shape[1]
        print(
            f"  metadata dim = {meta_dim} "
            f"(party {len(PARTY_VOCAB)} + credit {len(CREDIT_COLS)} + speaker/subject/context idx)"
        )
        print(f"  vocabs: {describe_vocabs(meta_vocabs)}")

        if "has_meta" in train_csv.columns:
            train_has = train_csv["has_meta"].astype(np.float32).values
            val_has   = val_csv["has_meta"].astype(np.float32).values
            test_has  = test_csv["has_meta"].astype(np.float32).values
            print(f"  has_meta:    train fraction = {train_has.mean():.3f}  "
                  f"val = {val_has.mean():.3f}  test = {test_has.mean():.3f}")
        else:
            train_has = np.ones(len(train_csv), dtype=np.float32)
            val_has   = np.ones(len(val_csv),   dtype=np.float32)
            test_has  = np.ones(len(test_csv),  dtype=np.float32)

    # ----- Datasets -----
    def ds(split, meta, has):
        return CoTDataset(
            features_npz=paths[f"feat_{split}"],
            masks_npz   =paths[f"mask_{split}"],
            meta=meta,
            has_meta=has,
        )
    train_ds = ds("train", train_meta, train_has)
    val_ds   = ds("val",   val_meta,   val_has)
    test_ds  = ds("test",  test_meta,  test_has)
    print(f"  train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
    print(
        f"  concept coverage train:  "
        f"E={int(train_ds.em.sum())}  M={int(train_ds.mo.sum())}  "
        f"N={int(train_ds.ne.sum())}  "
        f"({(train_ds.em.sum()+train_ds.mo.sum()+train_ds.ne.sum())/train_ds.em.numel()*100:.2f}% of subwords)"
    )
    if (train_ds.em.sum() + train_ds.mo.sum() + train_ds.ne.sum()) < 1000:
        print(
            "\n  !!!  Concept coverage is suspiciously low.\n"
            "       You probably need to re-run Phase 1 with the NEW step6:\n"
            "           python src/phase1/step6_concept_mapping.py\n"
            "           python src/phase1/step7_dataset_split.py\n"
            "       then DELETE data/concept_masks_*_L*.npz and re-run this script.\n"
        )

    sampler = None
    if balance:
        sampler = build_balanced_sampler(
            has_meta=train_has,
            labels=train_ds.labels.numpy(),
        )
        if sampler is None:
            print("  [balance] only one (domain, class) bucket present — falling back to shuffle.")
        else:
            print(f"  [balance] domain-balanced sampler enabled. "
                  f"Train buckets: {describe_buckets(train_has, train_ds.labels.numpy())}")

    if sampler is None:
        train_loader = DataLoader(train_ds, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=TRAIN_BATCH_SIZE, sampler=sampler)
    val_loader   = DataLoader(val_ds,   batch_size=TRAIN_BATCH_SIZE)
    test_loader  = DataLoader(test_ds,  batch_size=TRAIN_BATCH_SIZE)

    # ----- Model -----
    # When use_meta=True we pass meta_vocabs so CoTModel uses MetaEncoder
    # (party one-hot + credit z-scored + speaker/subject/context embeddings)
    model = CoTModel(
        bert_dim=768,
        hidden_size=192,
        num_lstm_layers=1,
        num_attn_heads=4,
        meta_dim=0,
        meta_vocabs=meta_vocabs,
        num_classes=2,
        dropout=TRAIN_DROPOUT,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  trainable params: {n_params:,}")

    # ----- Class weights + losses -----
    y_train_np = train_ds.labels.numpy()
    classes = np.unique(y_train_np)
    cw = compute_class_weight("balanced", classes=classes, y=y_train_np)
    cw_t = torch.tensor(cw, dtype=torch.float32, device=device)
    main_criterion = LabelSmoothingCE(smoothing=0.05, weight=cw_t)
    # aux_criterion is no longer used — see gated_aux_loss above.
    # Keeping the symbol available in case downstream scripts import it.
    aux_criterion  = nn.CrossEntropyLoss(weight=cw_t)  # noqa: F841

    # ----- Optimizer + warmup/decay -----
    lr = 5e-4
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=TRAIN_WEIGHT_DECAY
    )
    total_steps  = len(train_loader) * TRAIN_NUM_EPOCHS
    warmup_steps = max(1, int(0.1 * total_steps))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.1, 0.5 * (1 + np.cos(np.pi * progress)))  # cosine, floor 0.1

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ----- Logging -----
    tag = f"phase4_L{L}" if dataset == "liar" else f"phase4_{dataset}_L{L}"
    log_path = os.path.join(LOGS_DIR, f"{tag}_train.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch", "train_loss", "main_loss", "aux_loss", "cov_loss",
            "val_loss", "val_acc", "val_macro_f1",
            "em_mean", "mo_mean", "ne_mean", "lr",
        ])

    best_f1, best_state, patience = -1.0, None, 0
    LAMBDA_AUX = CONCEPT_AUX_LAMBDA
    LAMBDA_COV = 0.05

    for epoch in range(1, TRAIN_NUM_EPOCHS + 1):
        model.train()
        tot_loss = tot_main = tot_aux = tot_cov = 0.0
        n_batches = 0
        for batch in train_loader:
            (feats, amask, em_m, mo_m, ne_m, y, meta, has_meta) = _unpack(batch, use_meta, device)
            optimizer.zero_grad()
            (logits,
             em_s, mo_s, ne_s,
             em_aux, mo_aux, ne_aux,
             _, _, _,
             _) = model(feats, amask, em_m, mo_m, ne_m, meta, has_meta=has_meta)

            main_loss = main_criterion(logits, y)
            # Concept-presence-gated aux loss: each head's aux classifier
            # contributes only on examples where its concept actually appears.
            # Replaces the flat CE that produced the dead emotion head.
            aux_loss  = (
                gated_aux_loss(em_aux, y, em_m, weight=cw_t) +
                gated_aux_loss(mo_aux, y, mo_m, weight=cw_t) +
                gated_aux_loss(ne_aux, y, ne_m, weight=cw_t)
            )
            cov_loss  = coverage_penalty(em_s, em_m) \
                      + coverage_penalty(mo_s, mo_m) \
                      + coverage_penalty(ne_s, ne_m)

            loss = main_loss + LAMBDA_AUX * aux_loss + LAMBDA_COV * cov_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            scheduler.step()

            tot_loss += loss.item()
            tot_main += main_loss.item()
            tot_aux  += aux_loss.item()
            tot_cov  += cov_loss.item()
            n_batches += 1

        train_loss = tot_loss / n_batches
        avg_main   = tot_main / n_batches
        avg_aux    = tot_aux  / n_batches
        avg_cov    = tot_cov  / n_batches

        val = evaluate(model, val_loader, main_criterion, use_meta, device)
        cur_lr = optimizer.param_groups[0]["lr"]

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, train_loss, avg_main, avg_aux, avg_cov,
                val["loss"], val["acc"], val["macro_f1"],
                val["em_mean"], val["mo_mean"], val["ne_mean"], cur_lr,
            ])

        marker = ""
        if val["macro_f1"] > best_f1:
            best_f1 = val["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, os.path.join(MODELS_DIR, f"{tag}_best.pt"))
            patience = 0
            marker = " *"
        else:
            patience += 1

        print(
            f"ep {epoch:02d}  "
            f"train_loss={train_loss:.4f}  "
            f"val_acc={val['acc']:.4f}  val_f1={val['macro_f1']:.4f}  "
            f"E={val['em_mean']:.2f} M={val['mo_mean']:.2f} N={val['ne_mean']:.2f}  "
            f"lr={cur_lr:.2e}{marker}"
        )
        if patience >= EARLY_STOP_PATIENCE:
            print(f"  early stop @ epoch {epoch}")
            break

    # ----- Test -----
    model.load_state_dict(best_state)
    test = evaluate(model, test_loader, main_criterion, use_meta, device)
    print("\nTEST RESULTS")
    print(f"  acc       = {test['acc']:.4f}")
    print(f"  macro_f1  = {test['macro_f1']:.4f}")
    print(classification_report(
        test["labels"], test["preds"], target_names=["fake", "real"], digits=4
    ))

    breakdown = per_domain_breakdown(test["preds"], test["labels"], test["has_meta"])
    if breakdown and len(breakdown) > 1:
        print("PER-DOMAIN TEST METRICS")
        for name, m in breakdown.items():
            print(f"  {name:<5}  n={m['n']:>5}  acc={m['acc']:.4f}  macro_f1={m['macro_f1']:.4f}")

    # ----- Save rationale samples from test set -----
    model.eval()
    rat_dir = os.path.join(LOGS_DIR, "rationales_phase4")
    os.makedirs(rat_dir, exist_ok=True)
    with torch.no_grad():
        for batch in test_loader:
            (feats, amask, em_m, mo_m, ne_m, y, meta, has_meta) = _unpack(batch, use_meta, device)
            out = model(feats, amask, em_m, mo_m, ne_m, meta, has_meta=has_meta)
            logits, em_s, mo_s, ne_s = out[0], out[1], out[2], out[3]
            batch_rats = generate_batch_rationales(em_s, mo_s, ne_s, logits)
            out_path = os.path.join(rat_dir, f"sample_{tag}.txt")
            with open(out_path, "w") as f:
                for i, (rat, _, _) in enumerate(batch_rats[:10]):
                    f.write(f"--- example {i+1} ---\n{rat}\n\n")
            break
    print(f"  sample rationales -> {out_path}")

    # ----- Save final results -----
    results = {
        "phase": 4,
        "dataset": dataset,
        "L": L,
        "use_meta": use_meta,
        "seed": seed,
        "best_val_macro_f1": best_f1,
        "test_acc": test["acc"],
        "test_macro_f1": test["macro_f1"],
        "test_per_domain": breakdown,
        "n_params": n_params,
        "device": str(device),
        "em_mean": test["em_mean"],
        "mo_mean": test["mo_mean"],
        "ne_mean": test["ne_mean"],
    }
    with open(os.path.join(RESULTS_DIR, f"{tag}_seed{seed}.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"  results -> {RESULTS_DIR}/{tag}_seed{seed}.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--L", type=int, default=128)
    p.add_argument("--no-meta", action="store_true")
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument(
        "--dataset", choices=["liar", "merged"], default="liar",
        help="liar = LIAR only (default); merged = joint LIAR + CoAID corpus.",
    )
    p.add_argument(
        "--balance", action="store_true",
        help="Use domain-balanced sampler (LIAR/CoAID x fake/real). "
             "Recommended with --dataset merged.",
    )
    args = p.parse_args()
    main(L=args.L, use_meta=not args.no_meta, seed=args.seed,
         dataset=args.dataset, balance=args.balance)
