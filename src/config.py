"""
Central config for paths and pipeline constants.
Adjust LIAR_DATASET_DIR if the raw data lives elsewhere.
"""
import os

# -----------------------------
# DATASET PATHS
# -----------------------------
# Raw LIAR dataset directory (train.tsv, valid.tsv, test.tsv)
LIAR_DATASET_DIR = "/Users/sathvikreddysama/Downloads/liar_dataset"

# Project data directory for pipeline outputs (default: project/data)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Cross-domain datasets (auto-downloaded by data/download_datasets.py)
COAID_DIR     = os.path.join(DATA_DIR, "coaid")       # CoAID COVID-19 misinformation
LIAR_PLUS_DIR = os.path.join(DATA_DIR, "liar_plus")   # LIAR-PLUS with justifications

# -----------------------------
# SEQUENCE LENGTHS
# -----------------------------
SEQUENCE_LENGTHS = [128, 256, 512]
MAX_SEQUENCE_LEN = SEQUENCE_LENGTHS[0]

# -----------------------------
# VOCABULARY (legacy custom-vocab path; still used by some scripts)
# -----------------------------
VOCAB_MIN_FREQ = 2

# -----------------------------
# EMBEDDINGS  (Phase 2 Step 3)
# -----------------------------
# We now use a frozen DistilBERT encoder as a feature extractor.
# GloVe path is kept only for backward-compatibility; ignored if BERT_MODEL_NAME is set.
EMBEDDING_DIM = 768                     # DistilBERT hidden size
GLOVE_PATH    = None
BERT_MODEL_NAME = "distilbert-base-uncased"   # CPU/MPS-friendly, 66M params, 768-dim
BERT_MAX_LEN    = 128                          # cap subword sequence length for memory
BERT_BATCH_SIZE = 16                           # safe for 8GB unified memory on M2

# -----------------------------
# DEVICE
# -----------------------------
def get_device():
    """Return best available torch device: mps -> cuda -> cpu."""
    import torch
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# -----------------------------
# TRAINING DEFAULTS  (Phase 3 / 4)
# -----------------------------
TRAIN_BATCH_SIZE   = 32
TRAIN_LR           = 1e-3
TRAIN_WEIGHT_DECAY = 1e-5
TRAIN_DROPOUT      = 0.4
TRAIN_NUM_EPOCHS   = 30
EARLY_STOP_PATIENCE = 5
FOCAL_LOSS_GAMMA   = 2.0
GRAD_CLIP_NORM     = 1.0
RANDOM_SEED        = 42

# -----------------------------
# PHASE 4 CONCEPT REASONING
# -----------------------------
CONCEPT_AUX_LAMBDA = 0.3   # weight on auxiliary concept-prediction losses
CONCEPT_GATE_ALPHA = 1.0   # boost factor for concept-tagged tokens in attention

# -----------------------------
# OUTPUT DIRECTORIES
# -----------------------------
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR   = os.path.join(BASE_DIR, "logs")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
for _d in (DATA_DIR, MODELS_DIR, LOGS_DIR, RESULTS_DIR, COAID_DIR, LIAR_PLUS_DIR):
    os.makedirs(_d, exist_ok=True)
