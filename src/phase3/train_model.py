"""
Phase 3 — Train BiLSTM-Attention head on frozen DistilBERT features.

Adds vs the previous version:
  * Loads cached features (no recompute, no random embeddings)
  * Optional LIAR metadata branch (party one-hot + credit-history counts)
  * Focal loss (gamma=2) + balanced class weights
  * AdamW + warmup + gradient clipping
  * Early stopping on macro-F1
  * Logs both per-epoch metrics and final test-set numbers to logs/ + results/

Usage:
    python src/phase3/train_model.py [--L 128] [--no-meta] [--seed 42]
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
    DATA_DIR, MODELS_DIR, LOGS_DIR, RESULTS_DIR,
    TRAIN_BATCH_SIZE, TRAIN_LR, TRAIN_WEIGHT_DECAY, TRAIN_DROPOUT,
    TRAIN_NUM_EPOCHS, EARLY_STOP_PATIENCE, FOCAL_LOSS_GAMMA,
    GRAD_CLIP_NORM, RANDOM_SEED, get_device,
)
from bert_features import load_features
from meta_encoder import (
    PARTY_VOCAB, CREDIT_COLS, META_TOTAL_DIM,
    fit_metadata, encode_metadata, describe_vocabs,
)
from balanced_sampling import build_balanced_sampler, describe_buckets

sys.path.insert(0, os.path.join(_src, "phase3"))
from bilstm_attention import BiLSTMAttention


# ---------------------------------------------------------------------------
# Metadata builder (thin wrapper around meta_encoder so Phase 4 can still
# import `build_metadata` from here if it chooses to)
# ---------------------------------------------------------------------------
def build_metadata(
    df: pd.DataFrame,
    vocabs: dict | None = None,
    scaler: dict | None = None,
) -> tuple[np.ndarray, dict, dict]:
    """
    If vocabs/scaler are None, FIT on df (train-only semantics). Otherwise
    reuse them (val/test). Returns (meta (N, 15) float32, vocabs, scaler).
    """
    if vocabs is None or scaler is None:
        vocabs, scaler = fit_metadata(df)
    meta = encode_metadata(df, vocabs, scaler)
    return meta, vocabs, scaler


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class CachedFeatureDataset(Dataset):
    def __init__(
        self,
        npz_path: str,
        meta: np.ndarray | None = None,
        has_meta: np.ndarray | None = None,
    ):
        d = load_features(npz_path)
        self.features = torch.from_numpy(d["features"])
        self.attn     = torch.from_numpy(d["attention_mask"]).long()
        self.labels   = torch.from_numpy(d["labels"]).long()
        self.meta     = torch.from_numpy(meta).float() if meta is not None else None
        # has_meta is a (N,) {0,1} mask. When meta is disabled or every row has
        # metadata, it's left as None and the model treats every row as fully
        # metadata-bearing. We force it on for the merged LIAR + CoAID corpus.
        self.has_meta = (
            torch.from_numpy(has_meta).float() if has_meta is not None else None
        )
        assert len(self.features) == len(self.labels)
        if self.meta is not None:
            assert len(self.meta) == len(self.features), \
                f"meta {len(self.meta)} vs features {len(self.features)}"
        if self.has_meta is not None:
            assert len(self.has_meta) == len(self.features), \
                f"has_meta {len(self.has_meta)} vs features {len(self.features)}"

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        if self.meta is not None:
            hm = self.has_meta[i] if self.has_meta is not None else torch.tensor(1.0)
            return self.features[i], self.attn[i], self.meta[i], hm, self.labels[i]
        return self.features[i], self.attn[i], self.labels[i]


# ---------------------------------------------------------------------------
# Focal Loss (Lin et al. 2017) — handles mild class imbalance better than CE
# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()


# ---------------------------------------------------------------------------
# Train / Eval one epoch
# ---------------------------------------------------------------------------
def _step_batch(batch, use_meta, device):
    if use_meta:
        feats, mask, meta, has_meta, y = batch
        return (
            feats.to(device), mask.to(device),
            meta.to(device), has_meta.to(device),
            y.to(device),
        )
    feats, mask, y = batch
    return (feats.to(device), mask.to(device), None, None, y.to(device))


@torch.no_grad()
def evaluate(model, loader, criterion, use_meta, device):
    model.eval()
    losses, preds, ys, hms = [], [], [], []
    for batch in loader:
        feats, mask, meta, has_meta, y = _step_batch(batch, use_meta, device)
        logits, _, _ = model(feats, mask, meta, has_meta=has_meta)
        losses.append(criterion(logits, y).item())
        preds.append(logits.argmax(dim=-1).cpu().numpy())
        ys.append(y.cpu().numpy())
        if has_meta is not None:
            hms.append(has_meta.cpu().numpy())
    preds = np.concatenate(preds)
    ys = np.concatenate(ys)
    has_meta_arr = np.concatenate(hms) if hms else None
    return {
        "loss":     float(np.mean(losses)),
        "acc":      float(accuracy_score(ys, preds)),
        "macro_f1": float(f1_score(ys, preds, average="macro")),
        "preds":    preds,
        "labels":   ys,
        "has_meta": has_meta_arr,
    }


def per_domain_breakdown(preds, ys, has_meta_arr):
    """Split predictions by has_meta (1 = LIAR, 0 = CoAID) and compute per-domain metrics."""
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
    """Build all dataset-aware file paths in one place."""
    if dataset == "liar":
        feat_pre = "bert_features"
        split_pre = ""        # train_split_L128.csv
    else:                     # "merged" (or future: "coaid")
        feat_pre = f"bert_features_{dataset}"
        split_pre = f"{dataset}_"  # merged_train_split_L128.csv
    return {
        "feat_train": os.path.join(DATA_DIR, f"{feat_pre}_train_L{L}.npz"),
        "feat_val":   os.path.join(DATA_DIR, f"{feat_pre}_val_L{L}.npz"),
        "feat_test":  os.path.join(DATA_DIR, f"{feat_pre}_test_L{L}.npz"),
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
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = get_device()
    print(f"\nPhase 3 training (dataset={dataset}, L={L}, meta={use_meta}, seed={seed}, device={device})")

    # ----- Load cached BERT features -----
    paths = _paths(L, dataset)
    for p in (paths["feat_train"], paths["feat_val"], paths["feat_test"]):
        if not os.path.exists(p):
            hint = "python src/phase2/step3_embeddings.py"
            if dataset != "liar":
                hint += f" --dataset {dataset}"
            raise FileNotFoundError(
                f"Missing {p}. Run Phase 2 step 3 first:\n  {hint}"
            )

    # ----- Metadata (read split CSVs to align row order with features) -----
    meta_dim = 0
    meta_vocabs = None
    train_meta = val_meta = test_meta = None
    train_has = val_has = test_has = None
    if use_meta:
        train_csv = pd.read_csv(paths["csv_train"])
        val_csv   = pd.read_csv(paths["csv_val"])
        test_csv  = pd.read_csv(paths["csv_test"])
        # FIT on train only (no leakage), reuse on val/test
        # (fit_metadata internally restricts to has_meta == 1 if column present)
        train_meta, meta_vocabs, scaler = build_metadata(train_csv)
        val_meta,   _,           _      = build_metadata(val_csv,  meta_vocabs, scaler)
        test_meta,  _,           _      = build_metadata(test_csv, meta_vocabs, scaler)
        meta_dim = train_meta.shape[1]
        print(f"  metadata dim = {meta_dim} (party 7 + credit 5 + speaker/subject/context idx)")
        print(f"  vocabs: {describe_vocabs(meta_vocabs)}")

        # Per-row has_meta mask. Default = all 1s (LIAR), 0s for CoAID rows in merged.
        if "has_meta" in train_csv.columns:
            train_has = train_csv["has_meta"].astype(np.float32).values
            val_has   = val_csv["has_meta"].astype(np.float32).values
            test_has  = test_csv["has_meta"].astype(np.float32).values
            print(f"  has_meta:    train fraction with metadata = "
                  f"{train_has.mean():.3f}  val = {val_has.mean():.3f}  test = {test_has.mean():.3f}")
        else:
            train_has = np.ones(len(train_csv), dtype=np.float32)
            val_has   = np.ones(len(val_csv),   dtype=np.float32)
            test_has  = np.ones(len(test_csv),  dtype=np.float32)

    # ----- Datasets / loaders -----
    train_ds = CachedFeatureDataset(paths["feat_train"], train_meta, train_has)
    val_ds   = CachedFeatureDataset(paths["feat_val"],   val_meta,   val_has)
    test_ds  = CachedFeatureDataset(paths["feat_test"],  test_meta,  test_has)

    # Optional domain-balanced sampler for the joint LIAR + CoAID corpus.
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

    print(f"  train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

    # ----- Model -----
    # Smaller / single-layer head is a better fit for 9k training examples
    # and keeps the BiLSTM from oscillating at higher LRs.
    # When use_meta=True we pass meta_vocabs so the MetaEncoder handles
    # speaker/subject/context embeddings internally (ignores meta_dim path).
    model = BiLSTMAttention(
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

    # ----- Class weights + loss -----
    # Focal loss with gamma=2 was suppressing the gradient for already-easy
    # examples, leaving the BiLSTM stuck near chance. Weighted CE with mild
    # label smoothing is a more stable choice on LIAR.
    y_train_np = train_ds.labels.numpy()
    classes = np.unique(y_train_np)
    cw = compute_class_weight("balanced", classes=classes, y=y_train_np)
    cw_t = torch.tensor(cw, dtype=torch.float32, device=device)

    class SmoothedCE(nn.Module):
        def __init__(self, smoothing=0.05, weight=None):
            super().__init__(); self.s = smoothing; self.w = weight
        def forward(self, logits, y):
            n = logits.size(-1)
            with torch.no_grad():
                soft = torch.full_like(logits, self.s / (n - 1))
                soft.scatter_(1, y.unsqueeze(1), 1.0 - self.s)
            log_p = F.log_softmax(logits, dim=-1)
            loss = -(soft * log_p).sum(dim=-1)
            if self.w is not None:
                loss = loss * self.w[y]
            return loss.mean()

    criterion = SmoothedCE(smoothing=0.05, weight=cw_t)
    print(f"  class weights: {dict(zip(classes.tolist(), cw.tolist()))}")

    # ----- Optimizer + scheduler -----
    # Drop LR from 1e-3 to 5e-4 + cosine decay — the previous flat-linear
    # schedule swept past the optimum in epoch 3 and never recovered.
    base_lr = 5e-4
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=base_lr, weight_decay=TRAIN_WEIGHT_DECAY
    )
    total_steps = len(train_loader) * TRAIN_NUM_EPOCHS
    warmup_steps = max(1, int(0.1 * total_steps))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        # Cosine decay floored at 10% of base (keeps learning signal alive)
        return max(0.1, 0.5 * (1 + np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ----- Training loop -----
    tag = f"phase3_L{L}" if dataset == "liar" else f"phase3_{dataset}_L{L}"
    log_path = os.path.join(LOGS_DIR, f"{tag}_train.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "train_loss", "val_loss", "val_acc", "val_macro_f1", "lr"]
        )

    best_f1, best_state, patience = -1.0, None, 0
    for epoch in range(1, TRAIN_NUM_EPOCHS + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            feats, mask, meta, has_meta, y = _step_batch(batch, use_meta, device)
            optimizer.zero_grad()
            logits, _, _ = model(feats, mask, meta, has_meta=has_meta)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            scheduler.step()
            train_losses.append(loss.item())
        train_loss = float(np.mean(train_losses))

        val = evaluate(model, val_loader, criterion, use_meta, device)
        cur_lr = optimizer.param_groups[0]["lr"]

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch, train_loss, val["loss"], val["acc"], val["macro_f1"], cur_lr]
            )

        msg = (f"epoch {epoch:02d}  "
               f"train_loss={train_loss:.4f}  "
               f"val_loss={val['loss']:.4f}  "
               f"val_acc={val['acc']:.4f}  "
               f"val_f1={val['macro_f1']:.4f}  "
               f"lr={cur_lr:.2e}")
        if val["macro_f1"] > best_f1:
            best_f1 = val["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, os.path.join(MODELS_DIR, f"{tag}_best.pt"))
            patience = 0
            msg += "  *"
        else:
            patience += 1
        print(msg)
        if patience >= EARLY_STOP_PATIENCE:
            print(f"  early stop @ epoch {epoch} (no improvement for {patience} epochs)")
            break

    # ----- Test (best checkpoint) -----
    model.load_state_dict(best_state)
    test = evaluate(model, test_loader, criterion, use_meta, device)
    print("\nTEST RESULTS")
    print(f"  acc        = {test['acc']:.4f}")
    print(f"  macro_f1   = {test['macro_f1']:.4f}")
    print(classification_report(
        test["labels"], test["preds"], target_names=["fake", "real"], digits=4
    ))

    # Per-domain breakdown (only meaningful for the merged corpus where
    # has_meta varies — LIAR rows = 1, CoAID rows = 0).
    breakdown = per_domain_breakdown(test["preds"], test["labels"], test["has_meta"])
    if breakdown and len(breakdown) > 1:
        print("PER-DOMAIN TEST METRICS")
        for name, m in breakdown.items():
            print(f"  {name:<5}  n={m['n']:>5}  acc={m['acc']:.4f}  macro_f1={m['macro_f1']:.4f}")

    # ----- Save final results JSON -----
    results = {
        "phase": 3,
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
    }
    with open(os.path.join(RESULTS_DIR, f"{tag}_seed{seed}.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"  results -> {RESULTS_DIR}/{tag}_seed{seed}.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--L", type=int, default=128)
    p.add_argument("--no-meta", action="store_true", help="disable metadata branch")
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
