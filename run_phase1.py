"""
Run all Phase 1 steps (1-8) in order, then show a summary of outputs.
Usage: from project root, run:  python run_phase1.py
"""
import subprocess
import sys
import os

# Project root = directory containing this script
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(PROJECT_ROOT, "src")

PHASE1_DIR = os.path.join(SRC, "phase1")
STEPS = [
    ("Step 1 – Dataset selection + Cleaning", "step1_clean_liar.py"),
    ("Step 2 – Text normalization", "step2_normalize_text.py"),
    ("Step 3 – Tokenization", "step3_tokenize.py"),
    ("Step 4 – Sequence length handling", "step4_sequence_length.py"),
    ("Step 5 – Class balance analysis", "step5_class_balance.py"),
    ("Step 6 – Concept lexicon preparation", "step6_concept_mapping.py"),
    ("Step 7 – Dataset splitting", "step7_dataset_split.py"),
    ("Step 8 – Evaluation-oriented variants", "step8_evaluation_variants.py"),
]


def run_step(name, script):
    """Run a single step script; return True on success."""
    path = os.path.join(PHASE1_DIR, script)
    print(f"\n{'='*60}\n{name}\n{'='*60}")
    result = subprocess.run(
        [sys.executable, path],
        cwd=PROJECT_ROOT,
        capture_output=False,
    )
    return result.returncode == 0


def show_results():
    """List Phase 1 outputs and show a quick preview of key files."""
    sys.path.insert(0, SRC)
    from config import DATA_DIR

    print("\n" + "=" * 60)
    print("PHASE 1 RESULTS – Output files and preview")
    print("=" * 60)

    if not os.path.isdir(DATA_DIR):
        print(f"\nNo data directory yet: {DATA_DIR}")
        print("Run the pipeline first (steps 1–8).")
        return

    files = sorted(f for f in os.listdir(DATA_DIR) if not f.startswith("."))
    if not files:
        print(f"\nData directory is empty: {DATA_DIR}")
        return

    print(f"\nOutput directory: {DATA_DIR}")
    print(f"Files ({len(files)}):")
    for f in files:
        path = os.path.join(DATA_DIR, f)
        size = os.path.getsize(path) if os.path.isfile(path) else 0
        print(f"  • {f}  ({size:,} bytes)")

    # Preview key CSVs with pandas if available
    try:
        import pandas as pd
    except ImportError:
        print("\nInstall pandas to see table previews: pip install pandas")
        return

    previews = [
        ("liar_cleaned_step1.csv", ["statement", "label", "clean_statement"], "Step 1 cleaned"),
        ("liar_normalized_step2.csv", ["normalized_text", "label"], "Step 2 normalized"),
        ("train_split_L128.csv", ["statement", "binary_label"], "Train split (L=128)"),
        ("train_split_L256.csv", ["statement", "binary_label"], "Train split (L=256)"),
        ("train_split_L512.csv", ["statement", "binary_label"], "Train split (L=512)"),
    ]
    print("\n--- Preview of key outputs (L = 128, 256, 512 variants) ---")
    for filename, cols, title in previews:
        path = os.path.join(DATA_DIR, filename)
        if not os.path.isfile(path):
            continue
        try:
            df = pd.read_csv(path, nrows=100)
            available = [c for c in cols if c in df.columns]
            if not available:
                available = list(df.columns)[:3]
            sub = df[available].head(3)
            print(f"\n{title} ({filename})")
            print(f"  Shape: {df.shape[0]} rows (showing up to 100), {len(df.columns)} columns")
            print(sub.to_string(index=False))
        except Exception as e:
            print(f"\n{title}: could not preview – {e}")


def main():
    print("Phase 1 – Dataset and Preprocessing (run all 8 steps)")
    all_ok = True
    for name, script in STEPS:
        if not run_step(name, script):
            print(f"\nFailed: {script}")
            all_ok = False
            break

    if all_ok:
        show_results()
        print("\nPhase 1 finished successfully.")
    else:
        print("\nPhase 1 stopped due to an error.")
        sys.exit(1)


if __name__ == "__main__":
    main()
