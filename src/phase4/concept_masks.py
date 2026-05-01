"""
phase4/concept_masks.py  — REWORKED for BERT-subword alignment

Projects word-level concept tags (from Phase 1 step6) onto subword positions
(from the cached DistilBERT features in Phase 2 step3). Each split produces
three (N, T) float masks (T = BERT_MAX_LEN), aligned 1:1 with the features
file so Phase 4 training can do a simple dataset zip.

Tag set produced by Phase 1 step6:
    O, EMOTION, MODALITY, MODALITY_SCOPE, NEGATION, NEGATION_SCOPE

Collapsed to 3 concept channels here:
    EMOTION  channel : {EMOTION}
    MODALITY channel : {MODALITY, MODALITY_SCOPE}
    NEGATION channel : {NEGATION, NEGATION_SCOPE}

Why scope tokens are included:
    A modal cue like "might" only conveys hedging via the span it scopes
    over ("might have happened"). Tagging the cue alone would leave the
    concept-gated attention with a single-token attention budget — the
    head can't build a representation of "what is being hedged over".
    We therefore treat the cue and its dependency-parsed scope as the
    same concept channel.
"""
from __future__ import annotations

import ast
import os
import sys
import numpy as np
import pandas as pd

_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _src)
from bert_features import load_features, word_mask_to_subword_mask

# ---------------------------------------------------------------------------
# Channel → set of tags that activate it
# ---------------------------------------------------------------------------
CHANNEL_TAGS = {
    "emotion":  {"EMOTION"},
    "modality": {"MODALITY", "MODALITY_SCOPE"},
    "negation": {"NEGATION", "NEGATION_SCOPE"},
}


def _parse_tags(cell):
    if isinstance(cell, list):
        return cell
    if isinstance(cell, str):
        try:
            return ast.literal_eval(cell)
        except Exception:
            return []
    return []


def _word_mask(concept_tags, channel: str) -> np.ndarray:
    allowed = CHANNEL_TAGS[channel]
    return np.asarray([1 if t in allowed else 0 for t in concept_tags], dtype=np.int8)


def build_subword_masks_for_split(
    split_csv: str,
    features_npz: str,
    bert_max_len: int,
) -> dict:
    """
    Args:
        split_csv   : Phase 1 step7 CSV (e.g. train_split_L128.csv). Must
                      contain a `concept_tags` column (string-serialized list)
                      and must align row-for-row with features_npz.
        features_npz: Phase 2 step3 cache (e.g. bert_features_train_L128.npz).
                      Must contain `subword_to_word` of shape (N, bert_max_len).
        bert_max_len: length to project onto.

    Returns dict with keys {emotion, modality, negation, concept_tokens_per_row}
    where the first three are float32 arrays of shape (N, bert_max_len).
    """
    df = pd.read_csv(split_csv)
    if "concept_tags" not in df.columns:
        raise ValueError(f"{split_csv} has no concept_tags column — run Phase 1 step6+step7.")
    df["concept_tags"] = df["concept_tags"].apply(_parse_tags)

    feats = load_features(features_npz)
    sw2w = feats["subword_to_word"]                       # (N, T) int32
    N, T = sw2w.shape
    assert T == bert_max_len, f"bert_max_len mismatch: {T} vs {bert_max_len}"
    assert N == len(df), (
        f"Row count mismatch: csv={len(df)} vs npz={N}. "
        f"Did Phase 1 step7 and Phase 2 step3 run on the same inputs?"
    )

    emotion  = np.zeros((N, T), dtype=np.float32)
    modality = np.zeros((N, T), dtype=np.float32)
    negation = np.zeros((N, T), dtype=np.float32)

    for i in range(N):
        tags = df["concept_tags"].iloc[i]
        if not tags:
            continue
        em_word = _word_mask(tags, "emotion")
        mo_word = _word_mask(tags, "modality")
        ne_word = _word_mask(tags, "negation")
        emotion[i]  = word_mask_to_subword_mask(em_word, sw2w[i])
        modality[i] = word_mask_to_subword_mask(mo_word, sw2w[i])
        negation[i] = word_mask_to_subword_mask(ne_word, sw2w[i])

    return {
        "emotion":  emotion,
        "modality": modality,
        "negation": negation,
        "concept_tokens_per_row": (
            emotion.sum(axis=1) + modality.sum(axis=1) + negation.sum(axis=1)
        ).astype(np.int32),
    }


def cache_subword_masks(
    split_csv: str,
    features_npz: str,
    out_npz: str,
    bert_max_len: int,
    overwrite: bool = False,
):
    """Compute + cache subword masks for a split (saves ~(N, T)×3 float32).

    Auto-invalidates the cache if the existing file's row count or T-dim
    doesn't match the current features_npz / split_csv. This catches the
    case where the underlying corpus changed (e.g. CoAID news added to
    the merged splits) but the old mask file still exists.
    """
    if os.path.exists(out_npz) and not overwrite:
        try:
            existing = np.load(out_npz, allow_pickle=False)
            existing_n, existing_t = existing["emotion"].shape
            ref = np.load(features_npz, allow_pickle=False)
            ref_n, ref_t = ref["features"].shape[:2]
        except Exception as e:
            print(f"[concept_masks] could not validate {out_npz} ({e}) — rebuilding")
        else:
            if existing_n == ref_n and existing_t == ref_t and existing_t == bert_max_len:
                print(f"[concept_masks] cache exists, skipping: {out_npz}")
                return
            print(
                f"[concept_masks] cache stale: {out_npz} has shape "
                f"({existing_n}, {existing_t}) but features have ({ref_n}, {ref_t}) — rebuilding"
            )
    masks = build_subword_masks_for_split(split_csv, features_npz, bert_max_len)
    os.makedirs(os.path.dirname(out_npz), exist_ok=True)
    np.savez_compressed(out_npz, **masks)
    em, mo, ne = masks["emotion"], masks["modality"], masks["negation"]
    frac = (em.sum() + mo.sum() + ne.sum()) / (em.size)
    print(
        f"[concept_masks] wrote {out_npz}  "
        f"E={int(em.sum())} M={int(mo.sum())} N={int(ne.sum())}  "
        f"subword coverage = {frac*100:.2f}%"
    )


def load_subword_masks(out_npz: str) -> dict:
    d = np.load(out_npz, allow_pickle=False)
    return {k: d[k] for k in d.files}


if __name__ == "__main__":
    # Build + cache masks for all three splits at one L value.
    # Usage: python src/phase4/concept_masks.py [L]
    from config import DATA_DIR, BERT_MAX_LEN
    L = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    for split in ("train", "val", "test"):
        cache_subword_masks(
            split_csv=os.path.join(DATA_DIR, f"{split}_split_L{L}.csv"),
            features_npz=os.path.join(DATA_DIR, f"bert_features_{split}_L{L}.npz"),
            out_npz=os.path.join(DATA_DIR, f"concept_masks_{split}_L{L}.npz"),
            bert_max_len=BERT_MAX_LEN,
            overwrite=False,
        )
