"""
Phase 2 Step 3 — BERT Feature Pre-computation

Replaces the GloVe / random-init embedding matrix with frozen DistilBERT
contextual features cached to disk. Phase 3 / Phase 4 then load these cached
features and train a light head on top — no fine-tuning, CPU/MPS friendly.

For each L in SEQUENCE_LENGTHS we read the train/val/test split CSVs from
Phase 1 step 7 (which contain word-tokenised statements + concept tags +
metadata) and emit:

    data/bert_features_{split}_L{L}.npz
        features         (N, BERT_MAX_LEN, 768)  float32
        attention_mask   (N, BERT_MAX_LEN)       int8
        subword_to_word  (N, BERT_MAX_LEN)       int32
        n_words          (N,)                    int32
        labels           (N,)                    int32   (0=fake, 1=real)
        word_tokens      object array of original word lists (for rationale gen)

Run after Phase 1 + Phase 2 step 1/2.  Idempotent (skips cached files).
"""
from __future__ import annotations

import os
import sys
import ast
import pandas as pd
import numpy as np

_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _src)
from config import DATA_DIR, SEQUENCE_LENGTHS, BERT_MAX_LEN, BERT_MODEL_NAME
from bert_features import cache_features

LABEL_MAP = {"fake": 0, "real": 1}
SPLITS = ["train", "val", "test"]


def _load_split(L: int, split: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f"{split}_split_L{L}.csv")
    df = pd.read_csv(path)
    df["truncated_tokens"] = df["truncated_tokens"].apply(ast.literal_eval)
    return df


def main():
    print(f"\nPhase 2 Step 3 — caching frozen {BERT_MODEL_NAME} features")
    print(f"  output dir: {DATA_DIR}")
    print(f"  L values:   {SEQUENCE_LENGTHS}")
    print(f"  bert_max_len: {BERT_MAX_LEN}\n")

    for L in SEQUENCE_LENGTHS:
        # We use the same BERT_MAX_LEN regardless of word-truncation L for
        # apples-to-apples comparison across L (subword length is bounded the
        # same way). For L=128 word-truncated input, BERT_MAX_LEN=192 covers
        # the typical 1.3-1.5x subword expansion. For L=256/512, increase
        # BERT_MAX_LEN in config.py if you want full coverage.
        for split in SPLITS:
            df = _load_split(L, split)

            word_lists = df["truncated_tokens"].tolist()
            labels = np.array(
                [LABEL_MAP.get(str(b).lower(), 0) for b in df["binary_label"]],
                dtype=np.int32,
            )

            cache_path = os.path.join(DATA_DIR, f"bert_features_{split}_L{L}.npz")

            cache_features(
                word_lists=word_lists,
                cache_path=cache_path,
                bert_max_len=BERT_MAX_LEN,
                extras=dict(labels=labels),
                overwrite=False,
            )

    # Backward-compat stub: write a small embedding_params.txt so legacy code
    # that checks "pretrained" sees an honest True.
    params_path = os.path.join(DATA_DIR, "embedding_params.txt")
    with open(params_path, "w") as f:
        f.write(f"feature_extractor\t{BERT_MODEL_NAME}\n")
        f.write(f"frozen\tTrue\n")
        f.write(f"hidden_dim\t768\n")
        f.write(f"bert_max_len\t{BERT_MAX_LEN}\n")
    print(f"\n[step3] wrote {params_path}")
    print("[step3] done. Phase 3 train_model.py will load bert_features_*_L*.npz")


if __name__ == "__main__":
    main()
