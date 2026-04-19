"""
Phase 4 (fine-tuned) — Train the concept-gated CoT model end-to-end with
DistilBERT (top-N transformer layers unfrozen) instead of cached features.

What's new vs. train_phase4.py:
  * End-to-end fine-tuning (CoTModelFineTune)
  * Two-group LR: BERT @ 2e-5, head @ 5e-4
  * Linear warmup + linear decay (floored at 10% of base)
  * **Concept-presence-gated aux loss**: each head's auxiliary classifier
    only contributes loss on examples where that concept is present.
    This fixes the dead-emotion-head problem from the frozen pipeline.
  * **Attention-grounded rationale dump** at end of training: top-K tokens
    per head (decoded via tokenizer) + concept score + verdict.
  * On-the-fly tokenization + on-the-fly subword mask projection from
    cached `subword_to_word` indices — no need to regenerate npz caches.

Usage:
    python src/phase4/train_phase4_finetune.py [--L 128] [--unfreeze 2] [--no-meta]

Preconditions:
    Phase 1 step1..step7  (split CSVs with concept_tags)
    Phase 2 step3         (cached subword_to_word indices for mask projection)
"""
from __future__ import annotations

import os
import sys
import csv
import ast
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
    BERT_MODEL_NAME, BERT_MAX_LEN,
    TRAIN_WEIGHT_DECAY, TRAIN_NUM_EPOCHS, EARLY_STOP_PATIENCE,
    GRAD_CLIP_NORM, RANDOM_SEED, CONCEPT_AUX_LAMBDA, get_device,
)
from bert_features import load_features, word_mask_to_subword_mask
from meta_encoder import (
    PARTY_VOCAB, CREDIT_COLS, fit_metadata, encode_metadata, describe_vocabs,
)
sys.path.insert(0, os.path.join(_src, "phase3"))
from phase4.cot_finetune import CoTModelFineTune
from phase4.concept_masks import CHANNEL_TAGS, _word_mask, _parse_tags
from phase4.rationale import (
    generate_rationale_with_tokens, score_label, HIGH_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Dataset: input_ids + attention_mask + 3 subword concept masks + label + meta
# ---------------------------------------------------------------------------
class CoTFineTuneDataset(Dataset):
    def __init__(
        self,
        split_csv: str,
        features_npz: str,        # only used to read subword_to_word indices
        tokenizer,
        max_len: int,
        meta: np.ndarray | None = None,
    ):
        df = pd.read_csv(split_csv)
        if "concept_tags" not in df.columns:
            raise ValueError(
                f"{split_csv} has no concept_tags column — "
                f"run Phase 1 step6 + step7."
            )
        df["concept_tags"] = df["concept_tags"].apply(_parse_tags)

        text_col = "clean_statement" if "clean_statement" in df.columns else "statement"
        texts = df[text_col].fillna("").astype(str).tolist()
        labels = df["binary_label"].astype(int).tolist()

        # Tokenize once
        enc = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        self.input_ids      = enc["input_ids"].long()
        self.attention_mask = enc["attention_mask"].long()
        self.labels         = torch.tensor(labels, dtype=torch.long)

        # Subword-aligned concept masks via the same subword_to_word index used
        # by the frozen pipeline. We DON'T need the cached features themselves.
        feats = load_features(features_npz)
        sw2w = feats["subword_to_word"]
        N, T = sw2w.shape
        if N != len(df):
            raise RuntimeError(
                f"Row count mismatch: csv={len(df)} vs npz={N}. "
                f"Ensure Phase 1 step7 and Phase 2 step3 ran on the same inputs."
            )
        if T != max_len:
            raise RuntimeError(f"max_len mismatch: npz={T} vs requested={max_len}")

        em = np.zeros((N, T), dtype=np.float32)
        mo = np.zeros((N, T), dtype=np.float32)
        ne = np.zeros((N, T), dtype=np.float32)
        for i in range(N):
            tags = df["concept_tags"].iloc[i]
            if not tags:
                continue
            em[i] = word_mask_to_subword_mask(_word_mask(tags, "emotion"),  sw2w[i])
            mo[i] = word_mask_to_subword_mask(_word_mask(tags, "modality"), sw2w[i])
            ne[i] = word_mask_to_subword_mask(_word_mask(tags, "negation"), sw2w[i])
        self.em = torch.from_numpy(em)
        self.mo = torch.from_numpy(mo)
        self.ne = torch.from_numpy(ne)

        self.meta = torch.from_numpy(meta).float() if meta is not None else None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        base = (
            self.input_ids[i], self.attention_mask[i],
            self.em[i], self.mo[i], self.ne[i],
            self.labels[i],
        )
        if self.meta is not None:
            return base + (self.meta[i],)
        return base


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------
class LabelSmoothingCE(nn.Module):
    def __init__(self, smoothing: float = 0.05, weight: torch.Tensor | None = None):
        super().__init__(); self.smoothing = smoothing; self.weight = weight
    def forward(self, logits, targets):
        n = logits.size(-1)
        with torch.no_grad():
            soft = torch.full_like(logits, self.smoothing / (n - 1))
            soft.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        log_p = F.log_softmax(logits, dim=-1)
        loss = -(soft * log_p).sum(dim=-1)
        if self.weight is not None:
            loss = loss * self.weight[targets]
        return loss.mean()


def coverage_penalty(score: torch.Tensor, mask: torch.Tensor, target: float = 0.3) -> torch.Tensor:
    present = (mask.sum(dim=1) > 0).float()
    if present.sum() < 1:
        return torch.zeros((), device=score.device)
    mean_s = (score.squeeze(-1) * present).sum() / present.sum().clamp(min=1.0)
    return F.relu(target - mean_s)


def gated_aux_loss(aux_logits, targets, mask, weight=None):
    """
    Auxiliary CE that ONLY trains on examples where the concept is present.
    Fixes the dead-head problem (a head with concept_present=0 returns
    zero-context, so its aux classifier degenerates if trained on all rows).
    """
    present = (mask.sum(dim=1) > 0)
    if present.sum() < 1:
        return torch.zeros((), device=aux_logits.device)
    sub_logits = aux_logits[present]
    sub_targets = targets[present]
    return F.cross_entropy(sub_logits, sub_targets, weight=weight)


# ---------------------------------------------------------------------------
# Step / evaluate
# ---------------------------------------------------------------------------
def _unpack(batch, use_meta, device):
    if use_meta:
        ids, amask, em, mo, ne, y, meta = batch
        return (ids.to(device), amask.to(device),
                em.to(device), mo.to(device), ne.to(device),
                y.to(device), meta.to(device))
    ids, amask, em, mo, ne, y = batch
    return (ids.to(device), amask.to(device),
            em.to(device), mo.to(device), ne.to(device),
            y.to(device), None)


@torch.no_grad()
def evaluate(model, loader, main_criterion, use_meta, device):
    model.eval()
    losses, preds, ys = [], [], []
    em_avg, mo_avg, ne_avg = [], [], []
    for batch in loader:
        ids, amask, em_m, mo_m, ne_m, y, meta = _unpack(batch, use_meta, device)
        out = model(ids, amask, em_m, mo_m, ne_m, meta)
        logits, em_s, mo_s, ne_s = out[0], out[1], out[2], out[3]
        losses.append(main_criterion(logits, y).item())
        preds.append(logits.argmax(dim=-1).cpu().numpy())
        ys.append(y.cpu().numpy())
        em_avg.append(em_s.mean().item())
        mo_avg.append(mo_s.mean().item())
        ne_avg.append(ne_s.mean().item())
    preds = np.concatenate(preds); ys = np.concatenate(ys)
    return {
        "loss":     float(np.mean(losses)),
        "acc":      float(accuracy_score(ys, preds)),
        "macro_f1": float(f1_score(ys, preds, average="macro")),
        "em_mean":  float(np.mean(em_avg)),
        "mo_mean":  float(np.mean(mo_avg)),
        "ne_mean":  float(np.mean(ne_avg)),
        "preds":    preds,
        "labels":   ys,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(
    L: int = 128,
    unfreeze_top_n: int = 2,
    use_meta: bool = True,
    seed: int = RANDOM_SEED,
    batch_size: int = 16,
    grad_accum: int = 2,
    bert_lr: float = 2e-5,
    head_lr: float = 5e-4,
    epochs: int = TRAIN_NUM_EPOCHS,
    dropout: float = 0.2,
):
    torch.manual_seed(seed); np.random.seed(seed)
    device = get_device()
    print(
        f"\nPhase 4 FINE-TUNE training "
        f"(L={L}, unfreeze_top_n={unfreeze_top_n}, meta={use_meta}, "
        f"seed={seed}, device={device})"
    )

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)

    # ----- Metadata -----
    meta_vocabs = None
    train_meta = val_meta = test_meta = None
    if use_meta:
        train_csv_df = pd.read_csv(os.path.join(DATA_DIR, f"train_split_L{L}.csv"))
        val_csv_df   = pd.read_csv(os.path.join(DATA_DIR, f"val_split_L{L}.csv"))
        test_csv_df  = pd.read_csv(os.path.join(DATA_DIR, f"test_split_L{L}.csv"))
        meta_vocabs, scaler = fit_metadata(train_csv_df)
        train_meta = encode_metadata(train_csv_df, meta_vocabs, scaler)
        val_meta   = encode_metadata(val_csv_df,   meta_vocabs, scaler)
        test_meta  = encode_metadata(test_csv_df,  meta_vocabs, scaler)
        print(
            f"  metadata dim = {train_meta.shape[1]} "
            f"(party {len(PARTY_VOCAB)} + credit {len(CREDIT_COLS)} + speaker/subject/context idx)"
        )
        print(f"  vocabs: {describe_vocabs(meta_vocabs)}")

    # ----- Datasets -----
    def ds(split, meta):
        return CoTFineTuneDataset(
            split_csv=os.path.join(DATA_DIR, f"{split}_split_L{L}.csv"),
            features_npz=os.path.join(DATA_DIR, f"bert_features_{split}_L{L}.npz"),
            tokenizer=tokenizer,
            max_len=BERT_MAX_LEN,
            meta=meta,
        )
    train_ds = ds("train", train_meta)
    val_ds   = ds("val",   val_meta)
    test_ds  = ds("test",  test_meta)
    print(f"  train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
    print(
        f"  concept coverage train:  E={int(train_ds.em.sum())}  "
        f"M={int(train_ds.mo.sum())}  N={int(train_ds.ne.sum())}"
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size)

    # ----- Model -----
    model = CoTModelFineTune(
        bert_model_name=BERT_MODEL_NAME,
        unfreeze_top_n=unfreeze_top_n,
        hidden_size=192,
        num_lstm_layers=1,
        num_attn_heads=4,
        meta_dim=0,
        meta_vocabs=meta_vocabs,
        num_classes=2,
        dropout=dropout,
    ).to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  trainable params total: {n_trainable:,}")

    # ----- Class weights + losses -----
    y_train_np = train_ds.labels.numpy()
    classes = np.unique(y_train_np)
    cw = compute_class_weight("balanced", classes=classes, y=y_train_np)
    cw_t = torch.tensor(cw, dtype=torch.float32, device=device)
    main_criterion = LabelSmoothingCE(smoothing=0.05, weight=cw_t)

    # ----- Two-group optimizer -----
    optimizer = torch.optim.AdamW(
        [
            {"params": model.bert_parameters(), "lr": bert_lr, "weight_decay": 0.01},
            {"params": model.head_parameters(), "lr": head_lr, "weight_decay": TRAIN_WEIGHT_DECAY},
        ],
    )
    steps_per_epoch = max(1, len(train_loader) // grad_accum)
    total_steps = steps_per_epoch * epochs
    warmup = max(1, int(0.1 * total_steps))

    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return max(0.1, 1.0 - 0.9 * progress)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ----- Logging -----
    tag = f"phase4_finetune_L{L}_uf{unfreeze_top_n}"
    log_path = os.path.join(LOGS_DIR, f"{tag}.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch", "train_loss", "main_loss", "aux_loss", "cov_loss",
            "val_loss", "val_acc", "val_macro_f1",
            "em_mean", "mo_mean", "ne_mean", "bert_lr", "head_lr",
        ])

    LAMBDA_AUX = CONCEPT_AUX_LAMBDA
    LAMBDA_COV = 0.05
    best_f1, best_state, patience = -1.0, None, 0

    for epoch in range(1, epochs + 1):
        model.train()
        tot_loss = tot_main = tot_aux = tot_cov = 0.0
        n_batches = 0
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            ids, amask, em_m, mo_m, ne_m, y, meta = _unpack(batch, use_meta, device)
            (logits,
             em_s, mo_s, ne_s,
             em_aux, mo_aux, ne_aux,
             _, _, _,
             _) = model(ids, amask, em_m, mo_m, ne_m, meta)

            main_loss = main_criterion(logits, y)
            # Concept-presence-gated aux loss (THE fix for dead emotion head)
            aux_loss  = (
                gated_aux_loss(em_aux, y, em_m, weight=cw_t) +
                gated_aux_loss(mo_aux, y, mo_m, weight=cw_t) +
                gated_aux_loss(ne_aux, y, ne_m, weight=cw_t)
            )
            cov_loss  = (
                coverage_penalty(em_s, em_m) +
                coverage_penalty(mo_s, mo_m) +
                coverage_penalty(ne_s, ne_m)
            )

            loss = main_loss + LAMBDA_AUX * aux_loss + LAMBDA_COV * cov_loss
            (loss / grad_accum).backward()

            tot_loss += loss.item()
            tot_main += main_loss.item()
            tot_aux  += aux_loss.item()
            tot_cov  += cov_loss.item()
            n_batches += 1

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        train_loss = tot_loss / n_batches
        avg_main   = tot_main / n_batches
        avg_aux    = tot_aux  / n_batches
        avg_cov    = tot_cov  / n_batches

        val = evaluate(model, val_loader, main_criterion, use_meta, device)
        cur_bert_lr = optimizer.param_groups[0]["lr"]
        cur_head_lr = optimizer.param_groups[1]["lr"]

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, train_loss, avg_main, avg_aux, avg_cov,
                val["loss"], val["acc"], val["macro_f1"],
                val["em_mean"], val["mo_mean"], val["ne_mean"],
                cur_bert_lr, cur_head_lr,
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
            f"bert_lr={cur_bert_lr:.2e}{marker}"
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

    # ----- Attention-grounded rationales (NEW) -----
    rat_dir = os.path.join(LOGS_DIR, "rationales_phase4_finetune")
    os.makedirs(rat_dir, exist_ok=True)
    out_path = os.path.join(rat_dir, f"sample_{tag}.txt")
    model.eval()
    with torch.no_grad(), open(out_path, "w") as fout:
        n_written = 0
        for batch in test_loader:
            ids, amask, em_m, mo_m, ne_m, y, meta = _unpack(batch, use_meta, device)
            out = model(ids, amask, em_m, mo_m, ne_m, meta)
            (logits,
             em_s, mo_s, ne_s,
             _, _, _,
             em_alpha, mo_alpha, ne_alpha,
             _) = out
            B = ids.size(0)
            for i in range(B):
                if n_written >= 12:
                    break
                rationale = generate_rationale_with_tokens(
                    em_s[i].item(), mo_s[i].item(), ne_s[i].item(),
                    logits[i],
                    em_alpha[i], mo_alpha[i], ne_alpha[i],
                    ids[i], amask[i], tokenizer,
                    em_m[i], mo_m[i], ne_m[i],
                    top_k=5,
                )
                fout.write(f"--- example {n_written + 1} (gold={y[i].item()}) ---\n")
                fout.write(rationale + "\n\n")
                n_written += 1
            if n_written >= 12:
                break
    print(f"  attention-grounded rationales -> {out_path}")

    # ----- Save final results -----
    results = {
        "phase": 4,
        "variant": "finetune",
        "L": L,
        "unfreeze_top_n": unfreeze_top_n,
        "use_meta": use_meta,
        "seed": seed,
        "best_val_macro_f1": best_f1,
        "test_acc": test["acc"],
        "test_macro_f1": test["macro_f1"],
        "n_trainable_params": n_trainable,
        "device": str(device),
        "em_mean": test["em_mean"],
        "mo_mean": test["mo_mean"],
        "ne_mean": test["ne_mean"],
        "bert_lr": bert_lr,
        "head_lr": head_lr,
        "batch_size_eff": batch_size * grad_accum,
        "epochs_trained": epoch,
    }
    with open(os.path.join(RESULTS_DIR, f"{tag}_seed{seed}.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"  results -> {RESULTS_DIR}/{tag}_seed{seed}.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--L", type=int, default=128)
    p.add_argument("--unfreeze", type=int, default=2)
    p.add_argument("--no-meta", action="store_true")
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--accum", type=int, default=2)
    p.add_argument("--bert_lr", type=float, default=2e-5)
    p.add_argument("--head_lr", type=float, default=5e-4)
    p.add_argument("--epochs", type=int, default=TRAIN_NUM_EPOCHS)
    p.add_argument("--dropout", type=float, default=0.2)
    args = p.parse_args()
    main(
        L=args.L,
        unfreeze_top_n=args.unfreeze,
        use_meta=not args.no_meta,
        seed=args.seed,
        batch_size=args.batch,
        grad_accum=args.accum,
        bert_lr=args.bert_lr,
        head_lr=args.head_lr,
        epochs=args.epochs,
        dropout=args.dropout,
    )
