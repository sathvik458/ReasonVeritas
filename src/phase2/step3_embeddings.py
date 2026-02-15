"""
Phase 2 Step 3: Embedding Layer Setup
- Option A: Load GloVe, map vocab tokens to vectors; OOV/PAD → random or zero.
- Option B: Random initialization only (if GLOVE_PATH is None).
- Save embedding matrix for model: pretrained + fine-tunable during training.
"""
import os
import sys
import json
import numpy as np

_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _src)
from config import DATA_DIR, EMBEDDING_DIM, GLOVE_PATH

PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"

# -----------------------------
# LOAD VOCABULARY
# -----------------------------
vocab_path = os.path.join(DATA_DIR, "vocab_word2idx.json")
with open(vocab_path, "r", encoding="utf-8") as f:
    word2idx = json.load(f)
vocab_size = len(word2idx)
pad_id = word2idx[PAD_TOKEN]
unk_id = word2idx[UNK_TOKEN]

# -----------------------------
# LOAD GLOVE (if path set and file exists)
# -----------------------------
glove_vectors = {}
if GLOVE_PATH and os.path.isfile(GLOVE_PATH):
    print(f"Loading GloVe from {GLOVE_PATH} ...")
    with open(GLOVE_PATH, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip().split()
            if len(parts) < 2:
                continue
            word = parts[0]
            vec = np.array([float(x) for x in parts[1:]])
            if len(vec) != EMBEDDING_DIM:
                continue
            glove_vectors[word] = vec
    print(f"  Loaded {len(glove_vectors)} GloVe vectors (dim={EMBEDDING_DIM}).")
else:
    if GLOVE_PATH:
        print(f"GLOVE_PATH set but file not found; using random initialization.")
    else:
        print("GLOVE_PATH not set; using random initialization.")

# -----------------------------
# BUILD EMBEDDING MATRIX (vocab_size x EMBEDDING_DIM)
# -----------------------------
np.random.seed(42)
embedding_matrix = np.random.uniform(-0.1, 0.1, (vocab_size, EMBEDDING_DIM)).astype(np.float32)
embedding_matrix[pad_id] = 0.0  # PAD = zeros

matched = 0
for word, idx in word2idx.items():
    if word in (PAD_TOKEN, UNK_TOKEN):
        continue
    if word in glove_vectors:
        embedding_matrix[idx] = glove_vectors[word]
        matched += 1
# UNK and other OOV keep the random init (or could set UNK to zeros / mean of GloVe)

# -----------------------------
# SAVE
# -----------------------------
os.makedirs(DATA_DIR, exist_ok=True)
matrix_path = os.path.join(DATA_DIR, "embedding_matrix.npy")
np.save(matrix_path, embedding_matrix)

pretrained = bool(GLOVE_PATH and os.path.isfile(GLOVE_PATH))
params_path = os.path.join(DATA_DIR, "embedding_params.txt")
with open(params_path, "w") as f:
    f.write(f"vocab_size\t{vocab_size}\n")
    f.write(f"embedding_dim\t{EMBEDDING_DIM}\n")
    f.write(f"pretrained\t{pretrained}\n")
    f.write(f"glove_matched\t{matched}\n")

# -----------------------------
# SUMMARY
# -----------------------------
print("\nPhase 2 Step 3 – Embedding Layer")
print(f"Vocabulary size    : {vocab_size}")
print(f"Embedding dimension: {EMBEDDING_DIM}")
print(f"Pretrained (GloVe) : {pretrained}")
if glove_vectors:
    print(f"Vocab tokens in GloVe: {matched} / {vocab_size}")
print(f"Outputs:")
print(f"  - {matrix_path}")
print(f"  - {params_path}")
print("Use embedding_matrix.npy to initialize an embedding layer (fine-tune during training).")
