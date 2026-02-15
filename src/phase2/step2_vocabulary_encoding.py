"""
Phase 2 Step 2: Vocabulary Construction & Encoding
- Build vocabulary from training set only (reproducible, no leakage).
- Frequency threshold: keep tokens with freq >= VOCAB_MIN_FREQ; else <UNK>.
- Map tokens → integer indices; save word2idx and encoded train/val/test per L.
"""
import pandas as pd
import os
import sys
import ast
import json
import pickle
from collections import Counter
from sklearn.model_selection import train_test_split

_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _src)
from config import DATA_DIR, SEQUENCE_LENGTHS, VOCAB_MIN_FREQ

# Special tokens (must be in vocab)
PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"


def load_reasoning_tokens(path):
    """Load Phase 2 step 1 output and parse reasoning_tokens column."""
    df = pd.read_csv(path)
    if "reasoning_tokens" in df.columns:
        df["reasoning_tokens"] = df["reasoning_tokens"].apply(ast.literal_eval)
    return df


def build_vocab_from_tokens(token_lists, min_freq):
    """Build word2idx from token lists: keep token if count >= min_freq, else UNK."""
    counter = Counter()
    for tokens in token_lists:
        counter.update(tokens)
    # Reserve indices 0, 1 for PAD, UNK
    word2idx = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token, count in counter.most_common():
        if count >= min_freq and token not in word2idx:
            word2idx[token] = len(word2idx)
    return word2idx


def encode_tokens(tokens, word2idx, max_len):
    """Truncate to max_len and map tokens to indices; OOV → UNK."""
    unk_id = word2idx[UNK_TOKEN]
    pad_id = word2idx[PAD_TOKEN]
    indices = [word2idx.get(t, unk_id) for t in tokens[:max_len]]
    # Pad to max_len if needed
    while len(indices) < max_len:
        indices.append(pad_id)
    return indices


# -----------------------------
# LOAD PHASE 2 STEP 1 OUTPUT
# -----------------------------
input_path = os.path.join(DATA_DIR, "liar_phase2_reasoning_tokens.csv")
df = load_reasoning_tokens(input_path)

# Replicate Phase 1 split (70-15-15, stratified, same seed)
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

# -----------------------------
# BUILD VOCABULARY FROM TRAIN ONLY
# -----------------------------
word2idx = build_vocab_from_tokens(train_df["reasoning_tokens"].tolist(), VOCAB_MIN_FREQ)
idx2word = {v: k for k, v in word2idx.items()}
vocab_size = len(word2idx)

os.makedirs(DATA_DIR, exist_ok=True)
vocab_path = os.path.join(DATA_DIR, "vocab_word2idx.json")
with open(vocab_path, "w", encoding="utf-8") as f:
    json.dump(word2idx, f, ensure_ascii=False, indent=0)

params_path = os.path.join(DATA_DIR, "vocab_params.txt")
with open(params_path, "w") as f:
    f.write(f"vocab_size\t{vocab_size}\n")
    f.write(f"min_freq\t{VOCAB_MIN_FREQ}\n")
    f.write(f"pad_token\t{PAD_TOKEN}\n")
    f.write(f"unk_token\t{UNK_TOKEN}\n")

# -----------------------------
# ENCODE TRAIN / VAL / TEST FOR EACH L
# -----------------------------
def df_to_encoded(df, word2idx, L):
    indices = [encode_tokens(tokens, word2idx, L) for tokens in df["reasoning_tokens"]]
    labels = df["label"].tolist()
    binary = df["binary_label"].tolist()
    return {"indices": indices, "labels": labels, "binary_label": binary}

for L in SEQUENCE_LENGTHS:
    enc_train = df_to_encoded(train_df, word2idx, L)
    enc_val = df_to_encoded(val_df, word2idx, L)
    enc_test = df_to_encoded(test_df, word2idx, L)
    out = os.path.join(DATA_DIR, f"encoded_L{L}.pkl")
    with open(out, "wb") as f:
        pickle.dump({"train": enc_train, "val": enc_val, "test": enc_test}, f)

# -----------------------------
# SUMMARY
# -----------------------------
print("\nPhase 2 Step 2 – Vocabulary Construction & Encoding")
print(f"Vocabulary built from training set only (no leakage).")
print(f"Min frequency (keep token)  : {VOCAB_MIN_FREQ}")
print(f"Vocabulary size            : {vocab_size}")
print(f"Special tokens             : {PAD_TOKEN}=0, {UNK_TOKEN}=1")
print(f"Outputs:")
print(f"  - {vocab_path}")
print(f"  - {params_path}")
print(f"  - encoded_L128.pkl, encoded_L256.pkl, encoded_L512.pkl (train/val/test indices + labels)")
