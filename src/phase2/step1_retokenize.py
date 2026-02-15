"""
Phase 2 Step 1: Retokenization (Reasoning Units)
- Word-level tokenization (SpaCy)
- Preserve: punctuation tokens, negation, special tokens (<URL>, <NUM>, <ALLCAPS>/<allcaps>)
- Output: token sequences as atomic reasoning units for attention and vocabulary.
"""
import pandas as pd
import re
import os
import sys

_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _src)
from config import DATA_DIR

import spacy

# -----------------------------
# LOAD NORMALIZED DATA (Phase 1 Step 2)
# -----------------------------
input_path = os.path.join(DATA_DIR, "liar_normalized_step2.csv")
df = pd.read_csv(input_path)

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

# Special tokens that must stay as single units (after step 2 they may be lowercase)
SPECIAL_PATTERN = re.compile(r"<(?:URL|url|NUM|num|ALLCAPS|allcaps|[^>]+)>")
NEGATIONS = {"not", "no", "never", "none", "n't"}

log = {"total_tokens": 0, "punctuation_tokens": 0, "negation_tokens": 0, "special_tokens": 0}


def protect_special_tokens(text):
    """Replace <...> special tokens with placeholders so tokenizer won't split them."""
    placeholders = []
    def repl(m):
        placeholders.append(m.group(0))
        return f" __SPECIAL_{len(placeholders)-1}__ "
    protected = SPECIAL_PATTERN.sub(repl, str(text))
    return protected.strip(), placeholders


def restore_special_tokens(tokens, placeholders):
    """Restore original special tokens from placeholders."""
    out = []
    for t in tokens:
        match = re.match(r"__SPECIAL_(\d+)__", t)
        if match:
            idx = int(match.group(1))
            if idx < len(placeholders):
                out.append(placeholders[idx])
                log["special_tokens"] += 1
            else:
                out.append(t)
        else:
            out.append(t)
    return out


def tokenize_reasoning_units(text):
    """
    Tokenize into reasoning units:
    - Word-level (SpaCy)
    - Punctuation kept as separate tokens
    - Negation and special tokens preserved as single tokens
    """
    text = str(text)
    protected, placeholders = protect_special_tokens(text)
    doc = nlp(protected)
    tokens = [t.text for t in doc]
    tokens = restore_special_tokens(tokens, placeholders)

    for t in tokens:
        log["total_tokens"] += 1
        if t.lower() in NEGATIONS:
            log["negation_tokens"] += 1
        # Punctuation: non-alphanumeric and not our special token
        if re.match(r"^[\W_]+$", t) and not t.startswith("<") and not t.startswith("__"):
            log["punctuation_tokens"] += 1

    return tokens


# -----------------------------
# APPLY RETOKENIZATION
# -----------------------------
df["reasoning_tokens"] = df["normalized_text"].apply(tokenize_reasoning_units)
n = len(df)
avg_tokens = log["total_tokens"] / n if n else 0

# -----------------------------
# SAVE OUTPUT
# -----------------------------
os.makedirs(DATA_DIR, exist_ok=True)
output_path = os.path.join(DATA_DIR, "liar_phase2_reasoning_tokens.csv")
df.to_csv(output_path, index=False)

# -----------------------------
# SUMMARY
# -----------------------------
print("\nPhase 2 Step 1 – Retokenization (Reasoning Units)")
print(f"Total tokens           : {log['total_tokens']}")
print(f"Avg tokens per sample  : {avg_tokens:.2f}")
print(f"Punctuation tokens     : {log['punctuation_tokens']}")
print(f"Negation tokens        : {log['negation_tokens']}")
print(f"Special tokens (<URL>/<NUM>/<ALLCAPS>) : {log['special_tokens']}")
print(f"Output: {output_path}")
