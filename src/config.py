"""
Central config for paths and pipeline constants.
Adjust LIAR_DATASET_DIR if the raw data lives elsewhere.
"""
import os

# Raw Liar dataset directory (train.tsv, valid.tsv, test.tsv)
# NOTE Keep your dataset path here 
LIAR_DATASET_DIR = r"C:\Users\HP\Downloads\liar_dataset"  

# Project data directory for pipeline outputs (default: project/data)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Sequence length variants for ablation: L=128 vs 256 vs 512 (accuracy plateaus, reasoning faithfulness)
SEQUENCE_LENGTHS = [128, 256, 512]
# Default/primary L (e.g. for step5 class weights; first in list)
MAX_SEQUENCE_LEN = SEQUENCE_LENGTHS[0]

# Vocabulary: keep tokens with frequency >= VOCAB_MIN_FREQ; rest map to <UNK>
VOCAB_MIN_FREQ = 2

# Embeddings (Phase 2 step 3): dimension and optional GloVe path
EMBEDDING_DIM = 300
# Set to path of GloVe file (e.g. glove.6B.300d.txt) or None for random init only
GLOVE_PATH = None  # e.g. r"C:\Users\...\glove.6B.300d.txt"
