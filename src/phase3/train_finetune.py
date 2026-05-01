"""
Phase 3 (fine-tuned) — Train BiLSTM-Attention head on top of an
end-to-end fine-tuned DistilBERT (top-N layers unfrozen).

Differences vs. train_model.py (frozen):
  * Loads token IDs (re-tokenizes statements with DistilBertTokenizer once
    at dataset init), not cached BERT features.
  * Two-group AdamW: BERT params @ 2e-5, head params @ 5e-4.
  * Linear warmup (10%) + linear decay floored at 10% of base LR.
  * Lower dropout (0.2) — BERT itself is a strong regularizer.
  * Smaller batch (16) + gradient accumulation (2) → effective batch 32,
    fits comfortably on a 36 GB M3 Pro for unfreeze_top_n ≤ 4.

Usage:
    python src/phase3/train_finetune.py [--L 128] [--unfreeze 2] [--no-meta]

Run after:
    python src/phase1/step1_clean_liar.py    # ... up through step7
    python src/phase2/step3_embeddings.py    # frozen baseline (optional)
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
    BERT_MODEL_NAME, BERT_MAX_LEN,
    TRAIN_WEIGHT_DECAY, TRAIN_NUM_EPOCHS, EARLY_STOP_PATIENCE,
    GRAD_CLIP_NORM, RANDOM_SEED, get_device,
)
from balanced_sampling import build_balanced_sampler, describe_buckets
from meta_encoder import (
    PARTY_VOCAB, CREDIT_COLS, fit_metadata, encode_metadata, describe_vocabs,
)
sys.path.insert(0, os.path.join(_src, "phase3"))
from bert_finetune import FineTunedBERTClassifier


# ---------------------------------------------------------------------------
# Dataset: tokenize once at init, hold tensors in memory
# ---------------------------------------------------------------------------
class TextDataset(Dataset):
    def __init__(
        self,
        split_csv: str,
        tokenizer,
        max_len: int,
        meta: np.ndarray | None = None,
        has_meta: np.ndarray | None = None,
    ):
        df = pd.read_csv(split_csv)
        # Use the cleaned statement so we match Phase 1 normalization
        text_col = "clean_statement" if "clean_statement" in df.columns else "statement"
        texts = df[text_col].fillna("").astype(str).tolist()
        # binary_label may be a string ("fake"/"real") in merged splits.
        bl = df["binary_label"]
        # Handle both object/string types - check first value to determine if string labels
        first_val = bl.iloc[0] if len(bl) > 0 else None
        if first_val is not None and isinstance(first_val, str):
            labels = [0 if str(b).lower() == "fake" else 1 for b in bl]
        else:
            labels = bl.astype(int).tolist()

        enc = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        self.input_ids      = enc["input_ids"].long()                # (N, T)
        self.attention_mask = enc["attention_mask"].long()           # (N, T)
        self.labels         = torch.tensor(labels, dtype=torch.long) # (N,)
        self.meta           = torch.from_numpy(meta).float() if meta is not None else None
        self.has_meta       = (
            torch.from_numpy(has_meta).float() if has_meta is not None else None
        )
        if self.meta is not None:
            assert len(self.meta) == len(self.labels), \
                f"meta {len(self.meta)} vs labels {len(self.labels)}"

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        if self.meta is not None:
            hm = self.has_meta[i] if self.has_meta is not None else torch.tensor(1.0)
            return self.input_ids[i], self.attention_mask[i], self.meta[i], hm, self.labels[i]
        return self.input_ids[i], self.attention_mask[i], self.labels[i]


# ---------------------------------------------------------------------------
# Smoothed weighted CE (same as frozen pipeline — kept here for self-containment)
# ---------------------------------------------------------------------------
class SmoothedCE(nn.Module):
    def __init__(self, smoothing: float = 0.05, weight: torch.Tensor | None = None):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight

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


# ---------------------------------------------------------------------------
# Train / Eval helpers
# ---------------------------------------------------------------------------
def _step(batch, use_meta, device):
    if use_meta:
        ids, amask, meta, has_meta, y = batch
        return (ids.to(device), amask.to(device),
                meta.to(device), has_meta.to(device), y.to(device))
    ids, amask, y = batch
    return (ids.to(device), amask.to(device), None, None, y.to(device))


@torch.no_grad()
def evaluate(model, loader, criterion, use_meta, device):
    model.eval()
    losses, preds, ys, hms = [], [], [], []
    for batch in loader:
        ids, amask, meta, has_meta, y = _step(batch, use_meta, device)
        logits, _, _ = model(ids, amask, meta, has_meta=has_meta)
        losses.append(criterion(logits, y).item())
        preds.append(logits.argmax(dim=-1).cpu().numpy())
        ys.append(y.cpu().numpy())
        if has_meta is not None:
            hms.append(has_meta.cpu().numpy())
    preds = np.concatenate(preds); ys = np.concatenate(ys)
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
def _split_csv(L: int, dataset: str, split: str) -> str:
    if dataset == "liar":
        return os.path.join(DATA_DIR, f"{split}_split_L{L}.csv")
    return os.path.join(DATA_DIR, f"{dataset}_{split}_split_L{L}.csv")


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
    dataset: str = "liar",
    balance: bool = False,
):
    torch.manual_seed(seed); np.random.seed(seed)
    device = get_device()
    print(
        f"\nPhase 3 FINE-TUNE training "
        f"(dataset={dataset}, L={L}, unfreeze_top_n={unfreeze_top_n}, "
        f"meta={use_meta}, seed={seed}, device={device})"
    )

    # ----- Load tokenizer once -----
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)

    # ----- Metadata (fit on train only) -----
    meta_vocabs = None
    train_meta = val_meta = test_meta = None
    train_has = val_has = test_has = None
    if use_meta:
        train_csv_df = pd.read_csv(_split_csv(L, dataset, "train"))
        val_csv_df   = pd.read_csv(_split_csv(L, dataset, "val"))
        test_csv_df  = pd.read_csv(_split_csv(L, dataset, "test"))
        meta_vocabs, scaler = fit_metadata(train_csv_df)
        train_meta = encode_metadata(train_csv_df, meta_vocabs, scaler)
        val_meta   = encode_metadata(val_csv_df,   meta_vocabs, scaler)
        test_meta  = encode_metadata(test_csv_df,  meta_vocabs, scaler)
        print(
            f"  metadata dim = {train_meta.shape[1]} "
            f"(party {len(PARTY_VOCAB)} + credit {len(CREDIT_COLS)} + speaker/subject/context idx)"
        )
        print(f"  vocabs: {describe_vocabs(meta_vocabs)}")

        if "has_meta" in train_csv_df.columns:
            train_has = train_csv_df["has_meta"].astype(np.float32).values
            val_has   = val_csv_df["has_meta"].astype(np.float32).values
            test_has  = test_csv_df["has_meta"].astype(np.float32).values
            print(f"  has_meta:    train={train_has.mean():.3f}  "
                  f"val={val_has.mean():.3f}  test={test_has.mean():.3f}")
        else:
            train_has = np.ones(len(train_csv_df), dtype=np.float32)
            val_has   = np.ones(len(val_csv_df),   dtype=np.float32)
            test_has  = np.ones(len(test_csv_df),  dtype=np.float32)

    # ----- Datasets / loaders -----
    def ds(split, meta, has):
        return TextDataset(
            split_csv=_split_csv(L, dataset, split),
            tokenizer=tokenizer,
            max_len=BERT_MAX_LEN,
            meta=meta,
            has_meta=has,
        )
    train_ds = ds("train", train_meta, train_has)
    val_ds   = ds("val",   val_meta,   val_has)
    test_ds  = ds("test",  test_meta,  test_has)
    print(f"  train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}  batch={batch_size} (×{grad_accum} accum)")

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
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size)

    # ----- Model -----
    model = FineTunedBERTClassifier(
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

    # ----- Class weights + loss -----
    y_train_np = train_ds.labels.numpy()
    classes = np.unique(y_train_np)
    cw = compute_class_weight("balanced", classes=classes, y=y_train_np)
    cw_t = torch.tensor(cw, dtype=torch.float32, device=device)
    criterion = SmoothedCE(smoothing=0.05, weight=cw_t)

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
        return max(0.1, 1.0 - 0.9 * progress)        # linear decay floored at 0.1

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ----- Logging -----
    ds_suffix = "" if dataset == "liar" else f"_{dataset}"
    tag = f"phase3_finetune{ds_suffix}_L{L}_uf{unfreeze_top_n}"
    log_path = os.path.join(LOGS_DIR, f"{tag}.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "train_loss", "val_loss", "val_acc", "val_macro_f1", "bert_lr", "head_lr"]
        )

    best_f1, best_state, patience = -1.0, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            ids, amask, meta, has_meta, y = _step(batch, use_meta, device)
            logits, _, _ = model(ids, amask, meta, has_meta=has_meta)
            loss = criterion(logits, y) / grad_accum
            loss.backward()
            train_losses.append(loss.item() * grad_accum)
            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        train_loss = float(np.mean(train_losses))

        val = evaluate(model, val_loader, criterion, use_meta, device)
        cur_bert_lr = optimizer.param_groups[0]["lr"]
        cur_head_lr = optimizer.param_groups[1]["lr"]

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch, train_loss, val["loss"], val["acc"], val["macro_f1"],
                 cur_bert_lr, cur_head_lr]
            )

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
            f"bert_lr={cur_bert_lr:.2e}  head_lr={cur_head_lr:.2e}{marker}"
        )
        if patience >= EARLY_STOP_PATIENCE:
            print(f"  early stop @ epoch {epoch}")
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

    breakdown = per_domain_breakdown(test["preds"], test["labels"], test["has_meta"])
    if breakdown and len(breakdown) > 1:
        print("PER-DOMAIN TEST METRICS")
        for name, m in breakdown.items():
            print(f"  {name:<5}  n={m['n']:>5}  acc={m['acc']:.4f}  macro_f1={m['macro_f1']:.4f}")

    # ----- Save final results -----
    results = {
        "phase": 3,
        "variant": "finetune",
        "dataset": dataset,
        "L": L,
        "unfreeze_top_n": unfreeze_top_n,
        "use_meta": use_meta,
        "seed": seed,
        "test_per_domain": breakdown,
        "best_val_macro_f1": best_f1,
        "test_acc": test["acc"],
        "test_macro_f1": test["macro_f1"],
        "n_trainable_params": n_trainable,
        "device": str(device),
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
    p.add_argument("--unfreeze", type=int, default=2,
                   help="number of top DistilBERT layers to unfreeze (0..6)")
    p.add_argument("--no-meta", action="store_true")
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--accum", type=int, default=2)
    p.add_argument("--bert_lr", type=float, default=2e-5)
    p.add_argument("--head_lr", type=float, default=5e-4)
    p.add_argument("--epochs", type=int, default=TRAIN_NUM_EPOCHS)
    p.add_argument("--dropout", type=float, default=0.2)
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
        dataset=args.dataset,
        balance=args.balance,
    )
