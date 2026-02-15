"""
Show Phase 1 outputs: list files in data/ and preview key CSVs.
Run after Phase 1 has been executed. Usage:  python show_phase1_results.py
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC)

from config import DATA_DIR


def main():
    print("=" * 60)
    print("PHASE 1 – Results overview")
    print("=" * 60)

    if not os.path.isdir(DATA_DIR):
        print(f"\nData directory not found: {DATA_DIR}")
        print("Run Phase 1 first:  python run_phase1.py")
        return

    files = sorted(f for f in os.listdir(DATA_DIR) if not f.startswith("."))
    if not files:
        print(f"\nNo files in: {DATA_DIR}")
        return

    print(f"\nOutput directory: {DATA_DIR}")
    print(f"\nFiles ({len(files)}):")
    for f in files:
        path = os.path.join(DATA_DIR, f)
        if os.path.isfile(path):
            size = os.path.getsize(path)
            print(f"  • {f}  ({size:,} bytes)")
        else:
            print(f"  • {f}/")

    try:
        import pandas as pd
    except ImportError:
        print("\nInstall pandas to see table previews: pip install pandas")
        return

    # Preview important outputs (L = 128, 256, 512 variants)
    previews = [
        ("liar_cleaned_step1.csv", ["statement", "label", "clean_statement"], "Step 1 – Cleaned"),
        ("liar_normalized_step2.csv", ["normalized_text", "label"], "Step 2 – Normalized"),
        ("liar_tokenized_step3.csv", ["normalized_text", "tokens"], "Step 3 – Tokenized"),
        ("train_split_L128.csv", ["statement", "binary_label"], "Train split L=128"),
        ("train_split_L256.csv", ["statement", "binary_label"], "Train split L=256"),
        ("train_split_L512.csv", ["statement", "binary_label"], "Train split L=512"),
    ]

    print("\n" + "-" * 60)
    print("Preview of key outputs (first 3 rows)")
    print("-" * 60)

    for filename, cols, title in previews:
        path = os.path.join(DATA_DIR, filename)
        if not os.path.isfile(path):
            continue
        try:
            df = pd.read_csv(path, nrows=500)
            available = [c for c in cols if c in df.columns]
            if not available:
                available = list(df.columns)[:4]
            sub = df[available].head(3)
            # Truncate long text for display
            for c in sub.select_dtypes(include=["object"]).columns:
                sub = sub.copy()
                sub[c] = sub[c].astype(str).apply(lambda x: x[:60] + "..." if len(x) > 60 else x)
            print(f"\n{title}  [{filename}]")
            print(f"  Rows: {len(df)}, Columns: {list(df.columns)}")
            print(sub.to_string(index=False))
        except Exception as e:
            print(f"\n{title}: error – {e}")

    # Class weights if present
    weights_path = os.path.join(DATA_DIR, "class_weights.txt")
    if os.path.isfile(weights_path):
        print("\n" + "-" * 60)
        print("Class weights (class_weights.txt)")
        print("-" * 60)
        with open(weights_path) as f:
            print(f.read())


if __name__ == "__main__":
    main()
