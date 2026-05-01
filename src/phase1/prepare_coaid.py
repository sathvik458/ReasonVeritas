"""
src/phase1/prepare_coaid.py

End-to-end Phase 1 pipeline for CoAID (Cui & Lee 2020).

Now processes BOTH the Claim subset (short factual claims) and the News
subset (article headlines), combining them into a single CoAID corpus
that is column-compatible with the LIAR step6 output.

Mirrors what step1->step6 do for LIAR:
  1. Load all 16 snapshot CSVs from data/coaid/
       (Claim{Real,Fake}, News{Real,Fake} x 4 snapshots)
  2. For Claim files: use the `title` column (already a short statement).
     For News  files: also use `title` (the article headline) — short,
     LIAR-statement-like; we deliberately ignore the long `content` field
     to keep the input distribution close to LIAR + fit at L=128.
  3. Derive binary label from filename: "Real" -> "real", "Fake" -> "fake"
     (we write the binary label DIRECTLY, as a string, to avoid pandas
     auto-casting "true"/"false" to a bool dtype on CSV round-trip).
  4. Strip well-known label-leak prefixes from headlines
     ("FAKE:", "FALSE:", "DEBUNKED:", "TRUE:", "FACT-CHECK:", etc.) so the
     classifier can't shortcut on them.
  5. Clean / normalize / tokenize / truncate to L=128 / concept-tag
     (using the EXACT same `map_concepts` function used on LIAR).
  6. Dedup on the cleaned text (catches the same headline appearing
     across multiple snapshots).

Output: data/coaid_concepts_step6_L128.csv
        + new column `coaid_subset in {claim, news}` for diagnostics
        + has the SAME column schema as data/liar_concepts_step6_L128.csv
          (LIAR-only metadata cols filled with empty / 0).

Run:
    python data/download_datasets.py             # one-time CoAID fetch
    python src/phase1/prepare_coaid.py            # produces coaid_concepts_step6_L128.csv
    python src/phase1/merge_datasets.py           # joint LIAR + CoAID splits
"""
from __future__ import annotations

import os
import re
import sys
import glob
import unicodedata

import pandas as pd
import contractions
import spacy

_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _src)
from config import COAID_DIR, DATA_DIR  # noqa: E402

# Reuse the *exact* concept-mapping function used on LIAR — no drift between datasets.
sys.path.insert(0, os.path.join(_src, "phase1"))
from step6_concept_mapping import map_concepts  # noqa: E402


# ---------------------------------------------------------------------------
# Constants matching the LIAR pipeline
# ---------------------------------------------------------------------------
TARGET_L = 128
MIN_TOKENS = 3
DEDUP_KEY = "clean_statement"

# CoAID's per-row text column — both Claim*.csv and News*.csv use `title`.
CLAIM_TEXT_COL = "title"

# Label-leak prefixes sometimes appearing at the start of CoAID news headlines
# (e.g. "FAKE: 5G causes coronavirus"). We strip these *before* lowercasing so
# the classifier never sees the verdict baked into the input. Matched against
# a leading anchor with optional punctuation.
LEAK_PATTERNS = [
    r"^\s*fake\s*[:!\-]\s*",
    r"^\s*false\s*[:!\-]\s*",
    r"^\s*real\s*[:!\-]\s*",
    r"^\s*true\s*[:!\-]\s*",
    r"^\s*debunked\s*[:!\-]\s*",
    r"^\s*fact[-\s]?check(ed)?\s*[:!\-]\s*",
    r"^\s*hoax\s*[:!\-]\s*",
    r"^\s*myth\s*[:!\-]\s*",
    r"^\s*misinformation\s*[:!\-]\s*",
    r"^\s*misleading\s*[:!\-]\s*",
    r"^\s*claim(ed)?\s*[:!\-]\s*",
]
LEAK_RE = [re.compile(p, flags=re.IGNORECASE) for p in LEAK_PATTERNS]


def _strip_leak(text: str) -> str:
    """Remove leading verdict-like prefixes that would be label leaks."""
    s = str(text)
    # Iterate up to 2 times in case of double prefixes ("FAKE: FACT-CHECK: ...")
    for _ in range(2):
        new = s
        for r in LEAK_RE:
            new = r.sub("", new)
        if new == s:
            break
        s = new
    return s


# ---------------------------------------------------------------------------
# Step 1 / 2 equivalents — clean + normalize
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Match the regex set used in phase1/step1_clean_liar.clean_text."""
    text = _strip_leak(text)
    text = re.sub(r"http\S+|www\S+", "<URL>", text)
    text = re.sub(r"\b([A-Z]{2,})\b", r"<ALLCAPS> \1", text)
    text = re.sub(r"\b\d+\b", "<NUM>", text)
    text = re.sub(r"([!?]){2,}", r"\1", text)
    return text.lower().strip()


def normalize_text(text: str) -> str:
    """Match phase1/step2_normalize_text.normalize_text."""
    text = unicodedata.normalize("NFKC", str(text))
    text = contractions.fix(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Step 3 equivalent — SpaCy tokenization
# ---------------------------------------------------------------------------
try:
    NLP = spacy.load("en_core_web_sm", disable=["ner", "parser"])
except OSError:
    print("ERROR: spaCy model not found. Run:  python -m spacy download en_core_web_sm")
    raise


def tokenize(text: str) -> list[str]:
    return [t.text for t in NLP(text) if not t.is_space]


# ---------------------------------------------------------------------------
# Loader: read all CoAID Claim AND News CSVs
# ---------------------------------------------------------------------------
def _classify_filename(fname: str) -> tuple[str, str] | None:
    """
    Parse a CoAID filename into (subset, binary_label).
        ClaimRealCOVID-19.csv -> ("claim", "real")
        ClaimFakeCOVID-19.csv -> ("claim", "fake")
        NewsRealCOVID-19.csv  -> ("news",  "real")
        NewsFakeCOVID-19.csv  -> ("news",  "fake")
    Returns None for unknown patterns.
    """
    is_claim = "Claim" in fname
    is_news  = "News"  in fname
    is_fake  = "Fake"  in fname
    is_real  = "Real"  in fname

    if not (is_claim ^ is_news):    # exactly one of {claim, news}
        return None
    if not (is_fake ^ is_real):     # exactly one of {fake, real}
        return None
    subset = "claim" if is_claim else "news"
    label  = "fake"  if is_fake  else "real"
    return subset, label


def load_coaid(coaid_dir: str) -> pd.DataFrame:
    """
    Walk data/coaid/, find all Claim*COVID-19.csv AND News*COVID-19.csv,
    return a long DataFrame with [statement, label, coaid_subset].
    """
    rows = []
    files = sorted(glob.glob(os.path.join(coaid_dir, "*COVID-19.csv")))
    if not files:
        raise FileNotFoundError(
            f"No CoAID CSVs found under {coaid_dir}.\n"
            f"Run:  python data/download_datasets.py"
        )

    counts = {"claim_real": 0, "claim_fake": 0, "news_real": 0, "news_fake": 0}
    for fpath in files:
        fname = os.path.basename(fpath)
        parsed = _classify_filename(fname)
        if parsed is None:
            print(f"  [skip] unrecognized filename pattern: {fname}")
            continue
        subset, binary = parsed

        try:
            df = pd.read_csv(fpath)
        except Exception as e:
            print(f"  [fail] could not read {fname}: {e}")
            continue
        if CLAIM_TEXT_COL not in df.columns:
            print(f"  [warn] {fname} missing '{CLAIM_TEXT_COL}' column "
                  f"(has: {list(df.columns)[:6]}...) — skipping")
            continue

        df = df[[CLAIM_TEXT_COL]].rename(columns={CLAIM_TEXT_COL: "statement"})
        df["label"] = binary
        df["binary_label"] = binary
        df["coaid_subset"] = subset
        df["__source_file"] = fname
        rows.append(df)
        counts[f"{subset}_{binary}"] += len(df)
        print(f"  [load] {fname:<40} rows={len(df):>5}  subset={subset}  label={binary}")

    out = pd.concat(rows, ignore_index=True)
    out = out.dropna(subset=["statement"])
    out["statement"] = out["statement"].astype(str)

    print(
        "\n  per-(subset, label) raw counts: "
        f"claim_real={counts['claim_real']}  claim_fake={counts['claim_fake']}  "
        f"news_real={counts['news_real']}  news_fake={counts['news_fake']}"
    )
    return out


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    print(f"\n=== CoAID prep: {COAID_DIR} → {DATA_DIR}/coaid_concepts_step6_L{TARGET_L}.csv ===\n")

    # 1. Load + dedupe (claims AND news)
    df = load_coaid(COAID_DIR)
    n_raw = len(df)
    df = df.drop_duplicates(subset=["statement"]).reset_index(drop=True)
    print(f"\n  loaded={n_raw}  after literal-statement dedupe={len(df)}")

    # 2. Clean + normalize
    df["clean_statement"] = df["statement"].apply(clean_text)
    df["normalized_text"] = df["clean_statement"].apply(normalize_text)

    # 3. Final dedupe on cleaned text (catches near-duplicates across snapshots
    #    and across the claim<->news boundary, e.g. when a claim was extracted
    #    from a news headline).
    df = df.drop_duplicates(subset=[DEDUP_KEY]).reset_index(drop=True)
    print(f"  after content-dedupe={len(df)}")

    # 4. Tokenize
    df["tokens"] = df["normalized_text"].apply(tokenize)

    # 5. Truncate (head truncation, same as LIAR step4)
    df["truncated_tokens"] = df["tokens"].apply(lambda t: t[:TARGET_L])

    # 6. Drop near-empty rows (a few CoAID titles are 1-2 words like "Fact-check")
    pre = len(df)
    df = df[df["truncated_tokens"].apply(len) >= MIN_TOKENS].reset_index(drop=True)
    print(f"  dropped {pre - len(df)} rows with <{MIN_TOKENS} tokens")

    # 7. Concept tagging — *exact same function* as LIAR step6
    log = {
        "emotion_count": 0,
        "modality_count": 0, "modality_scope_count": 0,
        "negation_count": 0, "negation_scope_count": 0,
        "total_tokens": 0,
    }
    df["concept_tags"] = df["truncated_tokens"].apply(lambda t: map_concepts(t, log))

    # 8. Fill LIAR-only metadata columns with empty / zero so downstream
    #    code that expects the schema doesn't crash.
    df["subject"] = ""
    df["speaker"] = ""
    df["party"] = ""
    df["context"] = ""
    for col in ("barely_true_count", "false_count", "half_true_count",
                "mostly_true_count", "pants_fire_count"):
        df[col] = 0.0

    # 9. Reorder to match liar_concepts_step6_L128.csv column order, plus
    #    binary_label and coaid_subset for the merge step + diagnostics.
    col_order = [
        "label", "statement",
        "subject", "speaker", "party",
        "barely_true_count", "false_count", "half_true_count",
        "mostly_true_count", "pants_fire_count",
        "context",
        "clean_statement", "normalized_text",
        "tokens", "truncated_tokens", "concept_tags",
        "binary_label", "coaid_subset",
    ]
    df = df[col_order]

    out_path = os.path.join(DATA_DIR, f"coaid_concepts_step6_L{TARGET_L}.csv")
    df.to_csv(out_path, index=False)

    # 10. Summary
    tot = max(log["total_tokens"], 1)
    label_dist = df["binary_label"].value_counts().to_dict()
    subset_dist = df["coaid_subset"].value_counts().to_dict()
    cross = df.groupby(["coaid_subset", "binary_label"]).size().to_dict()
    print(f"\n  rows out: {len(df)}")
    print(f"  total tokens: {log['total_tokens']}")
    print(f"    EMOTION:        {log['emotion_count']:>5}  ({log['emotion_count']/tot*100:.2f}%)")
    print(f"    MODALITY (cue): {log['modality_count']:>5}  + scope {log['modality_scope_count']} "
          f"({(log['modality_count']+log['modality_scope_count'])/tot*100:.2f}%)")
    print(f"    NEGATION (cue): {log['negation_count']:>5}  + scope {log['negation_scope_count']} "
          f"({(log['negation_count']+log['negation_scope_count'])/tot*100:.2f}%)")
    print(f"\n  binary_label:  {label_dist}")
    print(f"  coaid_subset:  {subset_dist}")
    print(f"  (subset, label) cross: {cross}")
    print(f"\n  Wrote: {out_path}")


if __name__ == "__main__":
    main()
