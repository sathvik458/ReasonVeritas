"""
src/phase1/merge_datasets.py

Build the joint LIAR + CoAID training corpus.

Inputs:
    data/liar_concepts_step6_L128.csv     (from phase1 step1->step6)
    data/coaid_concepts_step6_L128.csv    (from phase1/prepare_coaid.py)

Output:
    data/merged_train_split_L128.csv
    data/merged_val_split_L128.csv
    data/merged_test_split_L128.csv

Key columns added by this script:
    dataset_source : "liar" | "coaid"
    has_meta       : 1 if LIAR (full speaker/party/credit metadata is real),
                     0 if CoAID (no metadata — the trainer zeros the meta branch)
    binary_label   : "real" | "fake"  (from LIAR's label or CoAID's filename)

Stratification:
    We stratify on the (binary_label, dataset_source) pair so each split has
    proportional fake/real coverage *and* proportional CoAID coverage. This
    matters because CoAID is much smaller than LIAR (~4k vs ~12.8k); without
    joint stratification a random 70/15/15 can put zero CoAID rows into val.

Run:
    python src/phase1/step1_clean_liar.py        # ... LIAR phase 1 must be done first ...
    python src/phase1/step6_concept_mapping.py
    python src/phase1/prepare_coaid.py
    python src/phase1/merge_datasets.py
"""
from __future__ import annotations

import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _src)
from config import DATA_DIR  # noqa: E402
from utils_logger import update_log  # noqa: E402


L = 128
RANDOM_SEED = 42

LIAR_PATH = os.path.join(DATA_DIR, f"liar_concepts_step6_L{L}.csv")
COAID_PATH = os.path.join(DATA_DIR, f"coaid_concepts_step6_L{L}.csv")


# ---------------------------------------------------------------------------
_FAKE_LABELS = {"false", "barely-true", "pants-fire", "fake"}


def _binarize(label) -> str:
    """
    LIAR step7's mapping: {false, barely-true, pants-fire} => fake; else real.
    Also accepts an already-binary 'fake'/'real' label (CoAID writes that).

    Robust to bool dtype: if pandas auto-cast 'true'/'false' strings to bool
    on read, str(False) = 'False' (capital F) which we lowercase before lookup.
    """
    s = str(label).strip().lower()
    return "fake" if s in _FAKE_LABELS else "real"


def _load_with_source(path: str, source: str, has_meta: int) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}. Run the upstream Phase 1 steps first.")
    df = pd.read_csv(path)
    df["dataset_source"] = source
    df["has_meta"] = has_meta
    # ALWAYS recompute binary_label — it's cheap and guards us against any
    # CSV round-trip dtype surprises (e.g. pandas turning 'true'/'false' into bool).
    df["binary_label"] = df["label"].apply(_binarize)
    return df


def main():
    print(f"\n=== merge_datasets: building LIAR + CoAID joint splits @ L={L} ===\n")
    liar = _load_with_source(LIAR_PATH, "liar", has_meta=1)
    coaid = _load_with_source(COAID_PATH, "coaid", has_meta=0)
    print(f"  LIAR rows : {len(liar):>6}  fake/real = "
          f"{(liar['binary_label']=='fake').sum()}/{(liar['binary_label']=='real').sum()}")
    print(f"  CoAID rows: {len(coaid):>6}  fake/real = "
          f"{(coaid['binary_label']=='fake').sum()}/{(coaid['binary_label']=='real').sum()}")

    # Align columns (CoAID prep already wrote them in the same order; this is a safety net)
    common_cols = [c for c in liar.columns if c in coaid.columns]
    if set(liar.columns) != set(coaid.columns):
        only_liar = set(liar.columns) - set(coaid.columns)
        only_coaid = set(coaid.columns) - set(liar.columns)
        if only_liar:
            print(f"  [warn] columns in LIAR not in CoAID: {sorted(only_liar)}")
        if only_coaid:
            print(f"  [warn] columns in CoAID not in LIAR: {sorted(only_coaid)}")
    merged = pd.concat([liar[common_cols], coaid[common_cols]], ignore_index=True)
    print(f"  merged rows: {len(merged)}")

    # Joint stratification key: combines class + dataset so splits stay balanced on both
    strat = merged["binary_label"].astype(str) + "_" + merged["dataset_source"].astype(str)

    train_df, temp_df = train_test_split(
        merged, test_size=0.30,
        stratify=strat, random_state=RANDOM_SEED,
    )
    temp_strat = temp_df["binary_label"].astype(str) + "_" + temp_df["dataset_source"].astype(str)
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50,
        stratify=temp_strat, random_state=RANDOM_SEED,
    )

    out_paths = {
        "train": os.path.join(DATA_DIR, f"merged_train_split_L{L}.csv"),
        "val":   os.path.join(DATA_DIR, f"merged_val_split_L{L}.csv"),
        "test":  os.path.join(DATA_DIR, f"merged_test_split_L{L}.csv"),
    }
    train_df.to_csv(out_paths["train"], index=False)
    val_df.to_csv(out_paths["val"], index=False)
    test_df.to_csv(out_paths["test"], index=False)

    def _summary(df, name):
        return {
            "n": len(df),
            "fake": int((df["binary_label"] == "fake").sum()),
            "real": int((df["binary_label"] == "real").sum()),
            "liar":  int((df["dataset_source"] == "liar").sum()),
            "coaid": int((df["dataset_source"] == "coaid").sum()),
        }

    summary = {
        "L": L,
        "split_ratio": "70/15/15 stratified on (binary_label, dataset_source)",
        "random_seed": RANDOM_SEED,
        "train": _summary(train_df, "train"),
        "val":   _summary(val_df, "val"),
        "test":  _summary(test_df, "test"),
    }
    update_log(f"MergeDatasets_L{L}", summary)

    for split_name, info in [("train", summary["train"]),
                             ("val",   summary["val"]),
                             ("test",  summary["test"])]:
        print(f"  [{split_name:>5}] n={info['n']:>5}  "
              f"fake/real={info['fake']}/{info['real']}  "
              f"liar/coaid={info['liar']}/{info['coaid']}  "
              f"-> {out_paths[split_name]}")

    print("\n  done.\n")


if __name__ == "__main__":
    main()
