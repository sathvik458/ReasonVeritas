import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split

_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _src)
from utils_logger import update_log
from config import DATA_DIR, SEQUENCE_LENGTHS


# -----------------------------
# SPLIT FOR EACH L (128, 256, 512) — SAME SEED SO SAME ROW ASSIGNMENT
# -----------------------------

os.makedirs(DATA_DIR, exist_ok=True)

for L in SEQUENCE_LENGTHS:
    input_path = os.path.join(DATA_DIR, f"liar_concepts_step6_L{L}.csv")
    df = pd.read_csv(input_path)

    if "binary_label" not in df.columns:
        fake_labels = ["false", "barely-true", "pants-fire"]
        df["binary_label"] = df["label"].apply(
            lambda x: "fake" if x in fake_labels else "real"
        )

    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["binary_label"], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["binary_label"], random_state=42
    )

    train_path = os.path.join(DATA_DIR, f"train_split_L{L}.csv")
    val_path = os.path.join(DATA_DIR, f"val_split_L{L}.csv")
    test_path = os.path.join(DATA_DIR, f"test_split_L{L}.csv")
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    train_counts = train_df["binary_label"].value_counts().to_dict()
    val_counts = val_df["binary_label"].value_counts().to_dict()
    test_counts = test_df["binary_label"].value_counts().to_dict()
    update_log(f"Step7_DatasetSplit_L{L}", {
        "dataset_name": "LIAR",
        "total_samples": len(df),
        "train_size": len(train_df),
        "validation_size": len(val_df),
        "test_size": len(test_df),
        "train_distribution": train_counts,
        "validation_distribution": val_counts,
        "test_distribution": test_counts,
        "random_seed": 42,
        "split_ratio": "70-15-15 (stratified)"
    })

print("\nDATASET SPLIT SUMMARY (L = 128, 256, 512)")
print(f"Train/val/test per L: train_split_L{{L}}.csv, val_split_L{{L}}.csv, test_split_L{{L}}.csv")
print(f"Same random_state=42 → identical row assignment across L for fair comparison.")
