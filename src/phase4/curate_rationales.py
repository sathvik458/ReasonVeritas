"""
src/phase4/curate_rationales.py

Curate a small, polished set of CoT-reasoning examples from the trained
Phase 4 model — formatted for the capstone viva.

We keep ONLY the examples where the model is doing real work:
  * prediction is correct (gold == pred)
  * confidence is high  (>= 0.70 by default)
  * at least one concept score is substantive (>= 0.15)
  * mix across (LIAR / CoAID) x (fake / real) so the panel sees variety

For each kept example we write a polished card:

    EXAMPLE 3  (LIAR · gold=fake · pred=fake · 0.81)
    Claim:  "..."
    Step 1 — EMOTION channel  (score=0.07, low)        top tokens: -none-
    Step 2 — MODALITY channel (score=0.31, moderate)   top tokens: "supports"(0.18), "demand"(0.11)
    Step 3 — NEGATION channel (score=0.04, low)        top tokens: -none-
    VERDICT: definite, unhedged claim attributed to a third party — predicted FAKE.

Run AFTER you have a Phase 4 checkpoint:

    python src/phase4/curate_rationales.py --tag phase4_merged_L128

Output:
    logs/rationales_phase4/curated_for_viva.txt
"""
from __future__ import annotations

import argparse
import os
import sys
import ast

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _src)
sys.path.insert(0, os.path.join(_src, "phase3"))
sys.path.insert(0, os.path.join(_src, "phase4"))

from config import (  # noqa: E402
    DATA_DIR, MODELS_DIR, LOGS_DIR, BERT_MODEL_NAME,
    TRAIN_BATCH_SIZE, TRAIN_DROPOUT, get_device,
)
from cot_model import CoTModel  # noqa: E402
from train_phase4 import CoTDataset  # noqa: E402
from meta_encoder import fit_metadata, encode_metadata  # noqa: E402


# ---------------------------------------------------------------------------
# Prose templates for each concept channel
# ---------------------------------------------------------------------------
SCORE_BAND = [
    (0.00, 0.10, "negligible"),
    (0.10, 0.20, "low"),
    (0.20, 0.40, "moderate"),
    (0.40, 0.65, "high"),
    (0.65, 1.01, "very high"),
]


def _band(score: float) -> str:
    for lo, hi, name in SCORE_BAND:
        if lo <= score < hi:
            return name
    return "very high"


def _human_summary(em: float, mo: float, ne: float, pred: int, conf: float) -> str:
    """A one-sentence reasoning summary using the channel scores."""
    label = "FAKE" if pred == 0 else "REAL"
    high = []
    if em >= 0.20: high.append(("affective framing", em))
    if mo >= 0.20: high.append(("modal/hedging language", mo))
    if ne >= 0.20: high.append(("negation cues", ne))
    high.sort(key=lambda x: -x[1])

    if not high:
        body = (
            "no concept channel produced a strong signal — the verdict was "
            "driven by the broader contextual representation"
        )
    elif len(high) == 1:
        body = f"primarily by the {high[0][0]} the model attended to"
    else:
        names = ", ".join(n for n, _ in high[:-1]) + f" and {high[-1][0]}"
        body = f"by {names}"

    return f"Verdict driven {body}. → predicted {label} (confidence {conf:.2f})."


# ---------------------------------------------------------------------------
# Path helpers (mirror the trainer's _paths)
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
        "feat_test":  os.path.join(DATA_DIR, f"{feat_pre}_test_L{L}.npz"),
        "mask_test":  os.path.join(DATA_DIR, f"{mask_pre}_test_L{L}.npz"),
        "csv_train":  os.path.join(DATA_DIR, f"{split_pre}train_split_L{L}.csv"),
        "csv_test":   os.path.join(DATA_DIR, f"{split_pre}test_split_L{L}.csv"),
    }


# ---------------------------------------------------------------------------
# Load top tokens for each concept channel
# ---------------------------------------------------------------------------
def _top_tokens_for_channel(
    alpha: np.ndarray,           # (T,) attention distribution
    attn_mask: np.ndarray,       # (T,) 1 = valid token
    sw2w: np.ndarray,            # (T,) subword -> source word index
    truncated_tokens: list[str],
    top_k: int = 4,
) -> list[tuple[str, float]]:
    """Return [(word, weight), ...] sorted by attention, dedup'd by source word."""
    scores = alpha.copy()
    scores[attn_mask == 0] = -np.inf
    if not np.isfinite(scores).any():
        return []
    order = np.argsort(-scores)
    seen: dict[str, float] = {}
    for pos in order:
        if scores[pos] <= 0 or not np.isfinite(scores[pos]):
            continue
        src = int(sw2w[pos])
        if src < 0 or src >= len(truncated_tokens):
            continue
        word = str(truncated_tokens[src]).strip()
        if not word or word in {"<num>", "<url>", "<allcaps>", "[CLS]", "[SEP]", "[PAD]"}:
            continue
        if word not in seen:
            seen[word] = float(alpha[pos])
        if len(seen) >= top_k:
            break
    return sorted(seen.items(), key=lambda x: -x[1])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(
    tag: str,
    L: int = 128,
    dataset: str = "merged",
    n_examples: int = 8,
    min_confidence: float = 0.70,
    min_concept_score: float = 0.15,
    seed: int = 42,
):
    device = get_device()
    paths = _paths(L, dataset)

    # ----- Test data + meta -----
    train_csv_df = pd.read_csv(paths["csv_train"])
    test_csv_df  = pd.read_csv(paths["csv_test"]).reset_index(drop=True)
    meta_vocabs, scaler = fit_metadata(train_csv_df)
    test_meta = encode_metadata(test_csv_df, meta_vocabs, scaler)
    test_has  = (
        test_csv_df["has_meta"].astype(np.float32).values
        if "has_meta" in test_csv_df.columns
        else np.ones(len(test_csv_df), dtype=np.float32)
    )

    # Source word lists (for decoding subwords back to readable tokens)
    test_csv_df["__tokens"] = test_csv_df["truncated_tokens"].apply(
        lambda x: x if isinstance(x, list) else ast.literal_eval(str(x))
    )

    # subword_to_word alignment from the BERT feature cache
    feats_npz = np.load(paths["feat_test"], allow_pickle=False)
    sw2w_all = feats_npz["subword_to_word"]    # (N, T)

    test_ds = CoTDataset(
        features_npz=paths["feat_test"],
        masks_npz   =paths["mask_test"],
        meta=test_meta,
        has_meta=test_has,
    )
    loader = DataLoader(test_ds, batch_size=TRAIN_BATCH_SIZE)

    # ----- Model -----
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
    ckpt = os.path.join(MODELS_DIR, f"{tag}_best.pt")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"No checkpoint at {ckpt}")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    print(f"\n[curate] {ckpt}  device={device}  test rows={len(test_ds)}")

    # ----- Run model and collect per-row data -----
    rows: list[dict] = []
    row_offset = 0
    with torch.no_grad():
        for batch in loader:
            feats, amask, em_m, mo_m, ne_m, y, meta, has_meta = (
                t.to(device) if isinstance(t, torch.Tensor) else t for t in batch
            )
            has_meta = has_meta.float()
            out = model(feats, amask, em_m, mo_m, ne_m, meta, has_meta=has_meta)
            logits = out[0]
            em_score, mo_score, ne_score = out[1], out[2], out[3]
            em_alpha, mo_alpha, ne_alpha = out[7], out[8], out[9]

            probs = torch.softmax(logits, dim=-1)
            preds = probs.argmax(dim=-1)
            conf = probs.gather(1, preds.unsqueeze(-1)).squeeze(-1)

            B = feats.size(0)
            for i in range(B):
                global_i = row_offset + i
                rows.append({
                    "row_index": global_i,
                    "label":     int(y[i].item()),
                    "pred":      int(preds[i].item()),
                    "conf":      float(conf[i].item()),
                    "em":        float(em_score[i].item()),
                    "mo":        float(mo_score[i].item()),
                    "ne":        float(ne_score[i].item()),
                    "em_alpha":  em_alpha[i].cpu().numpy(),
                    "mo_alpha":  mo_alpha[i].cpu().numpy(),
                    "ne_alpha":  ne_alpha[i].cpu().numpy(),
                    "attn_mask": amask[i].cpu().numpy(),
                })
            row_offset += B

    # ----- Filter to "good" examples -----
    def is_good(r):
        if r["pred"] != r["label"]:
            return False
        if r["conf"] < min_confidence:
            return False
        if max(r["em"], r["mo"], r["ne"]) < min_concept_score:
            return False
        return True

    keep_pool = [r for r in rows if is_good(r)]
    print(f"  {len(keep_pool)}/{len(rows)} examples pass filters "
          f"(correct, conf≥{min_confidence}, max-concept≥{min_concept_score})")

    # ----- Stratify across (domain, class) -----
    # has_meta column tells us domain
    test_csv_df["__has_meta"] = test_csv_df.get("has_meta", 1)
    buckets = {("liar", 0): [], ("liar", 1): [], ("coaid", 0): [], ("coaid", 1): []}
    for r in keep_pool:
        ds_src = "liar" if int(test_csv_df.iloc[r["row_index"]].get("__has_meta", 1)) == 1 else "coaid"
        buckets[(ds_src, r["label"])].append(r)

    # Sort each bucket by descending max-concept-score so we surface the
    # examples where the CoT story is most visually obvious.
    for k in buckets:
        buckets[k].sort(key=lambda r: -max(r["em"], r["mo"], r["ne"]))

    # Round-robin pick across buckets until we have n_examples
    chosen: list[tuple[str, dict]] = []
    order = [("liar", 0), ("liar", 1), ("coaid", 0), ("coaid", 1)]
    while len(chosen) < n_examples:
        progressed = False
        for k in order:
            if len(chosen) >= n_examples:
                break
            if buckets[k]:
                r = buckets[k].pop(0)
                ds_src, _ = k
                chosen.append((ds_src, r))
                progressed = True
        if not progressed:
            break
    print(f"  selected {len(chosen)} examples for the viva file")
    if len(chosen) < n_examples:
        print(f"  (wanted {n_examples}; some buckets ran dry — that's fine)")

    # ----- Write the polished viva file -----
    out_path = os.path.join(LOGS_DIR, "rationales_phase4", "curated_for_viva.txt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    LABELS = {0: "fake", 1: "real"}

    HEADER = (
        "═" * 72 + "\n"
        "Curated CoT reasoning examples — Phase 4 (concept-gated CoT)\n"
        f"Source: models/{tag}_best.pt\n"
        "Filters: prediction correct, confidence ≥ "
        f"{min_confidence}, ≥1 concept score ≥ {min_concept_score}\n"
        "═" * 72 + "\n\n"
    )

    with open(out_path, "w") as fh:
        fh.write(HEADER)
        for idx, (ds_src, r) in enumerate(chosen, start=1):
            row = test_csv_df.iloc[r["row_index"]]
            claim = str(row.get("clean_statement", row.get("statement", "")))
            tokens = row["__tokens"]
            sw2w = sw2w_all[r["row_index"]]

            em_top = _top_tokens_for_channel(r["em_alpha"], r["attn_mask"], sw2w, tokens)
            mo_top = _top_tokens_for_channel(r["mo_alpha"], r["attn_mask"], sw2w, tokens)
            ne_top = _top_tokens_for_channel(r["ne_alpha"], r["attn_mask"], sw2w, tokens)

            def _fmt_tokens(top):
                return ", ".join(f"\"{w}\" ({s:.2f})" for w, s in top) if top else "—"

            fh.write(
                f"───────────────────────────────────────────────────────────────────────\n"
                f"EXAMPLE {idx}  ({ds_src.upper()} · gold={LABELS[r['label']]} · "
                f"pred={LABELS[r['pred']]} · conf={r['conf']:.2f})\n"
                f"───────────────────────────────────────────────────────────────────────\n"
                f"Claim:\n  \"{claim}\"\n\n"
                f"Reasoning trace:\n"
                f"  Step 1 — EMOTION channel   "
                f"(score={r['em']:.2f}, {_band(r['em'])})\n"
                f"           top tokens: {_fmt_tokens(em_top)}\n\n"
                f"  Step 2 — MODALITY channel  "
                f"(score={r['mo']:.2f}, {_band(r['mo'])})\n"
                f"           top tokens: {_fmt_tokens(mo_top)}\n\n"
                f"  Step 3 — NEGATION channel  "
                f"(score={r['ne']:.2f}, {_band(r['ne'])})\n"
                f"           top tokens: {_fmt_tokens(ne_top)}\n\n"
                f"  Conclusion:\n"
                f"    {_human_summary(r['em'], r['mo'], r['ne'], r['pred'], r['conf'])}\n"
                f"\n"
            )

    print(f"\n  wrote {out_path}")
    print("\nUse this file in the viva — show 2-3 cards on screen, "
          "explain Step 1→2→3 then the conclusion line.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--tag", default="phase4_merged_L128",
        help="Checkpoint prefix in models/, e.g. phase4_merged_L128 or phase4_L128.",
    )
    p.add_argument("--L", type=int, default=128)
    p.add_argument("--dataset", choices=["liar", "merged"], default="merged")
    p.add_argument("--n_examples", type=int, default=8)
    p.add_argument("--min_confidence", type=float, default=0.70)
    p.add_argument("--min_concept_score", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    main(
        tag=args.tag, L=args.L, dataset=args.dataset,
        n_examples=args.n_examples,
        min_confidence=args.min_confidence,
        min_concept_score=args.min_concept_score,
        seed=args.seed,
    )
