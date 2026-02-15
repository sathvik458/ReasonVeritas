# Cater-polar

**Confidence-Calibrated Chain-of-Thought Reasoning for Interpretable and Reliable Deep Learning Models**  
*(Liar dataset for fake-news / claim prediction)*

---

## Overview

Cater-polar is a pipeline to preprocess the Liar dataset and prepare representations for interpretable, confidence-aware fake-news prediction. It is organized in **phases**: Phase 1 (dataset and preprocessing), Phase 2 (representation and encoding), with later phases for the base model, concept-level reasoning, and confidence/calibration.

- **Phase 1:** Clean → normalize → tokenize → truncate (L=128, 256, 512) → class balance → concept mapping → stratified splits → evaluation variants.
- **Phase 2:** Retokenization (reasoning units) → vocabulary & encoding → embedding matrix (GloVe or random init).

All pipeline outputs are written to the project `data/` directory. Paths and constants are centralized in `src/config.py`.

---

## Directory Structure

```
Cater-polar/
├── README.md
├── run_phase1.py              # Run all Phase 1 steps and show summary
├── show_phase1_results.py     # View Phase 1 outputs without re-running
├── requirements.txt
├── data/                      # All pipeline outputs (created on first run)
│   ├── liar_cleaned_step1.csv
│   ├── liar_normalized_step2.csv
│   ├── liar_tokenized_step3.csv
│   ├── liar_truncated_L128.csv, liar_truncated_L256.csv, liar_truncated_L512.csv
│   ├── class_weights.txt
│   ├── liar_concepts_step6_L128.csv, _L256.csv, _L512.csv
│   ├── train_split_L128.csv, val_split_L128.csv, test_split_L128.csv  (and L256, L512)
│   ├── train_eval_variants_step8_L128.csv, _L256.csv, _L512.csv
│   ├── liar_phase2_reasoning_tokens.csv   # Phase 2 step 1
│   ├── vocab_word2idx.json, vocab_params.txt   # Phase 2 step 2
│   ├── encoded_L128.pkl, encoded_L256.pkl, encoded_L512.pkl   # Phase 2 step 2
│   └── embedding_matrix.npy, embedding_params.txt            # Phase 2 step 3
├── logs/                      # Experiment logs (from utils_logger)
└── src/
    ├── config.py              # Paths and constants (edit here for your setup)
    ├── utils_logger.py
    ├── phase1/                # Phase 1: Dataset and Preprocessing
    │   ├── step1_clean_liar.py
    │   ├── step2_normalize_text.py
    │   ├── step3_tokenize.py
    │   ├── step4_sequence_length.py
    │   ├── step5_class_balance.py
    │   ├── step6_concept_mapping.py
    │   ├── step7_dataset_split.py
    │   └── step8_evaluation_variants.py
    └── phase2/                # Phase 2: Representation and Encoding
        ├── step1_retokenize.py
        ├── step2_vocabulary_encoding.py
        └── step3_embeddings.py
```

---

## Prerequisites

- **Python** 3.8+ ([python.org](https://www.python.org/downloads/))
- **Liar dataset:** place `train.tsv`, `valid.tsv`, and `test.tsv` in a folder and set that path in `src/config.py` (see Configuration below).

---

## Installation

From the project root:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

**Main dependencies:** `pandas`, `numpy`, `scikit-learn`, `spacy`, `emoji`, `beautifulsoup4`, `contractions`.

---

## Configuration

Edit **`src/config.py`** before running:

| Variable | Meaning | Example |
|----------|---------|---------|
| `LIAR_DATASET_DIR` | Folder containing raw `train.tsv`, `valid.tsv`, `test.tsv` | `r"C:\Users\...\liar_dataset"` |
| `DATA_DIR` | Where all pipeline outputs are saved (default: project `data/`) | `os.path.join(BASE_DIR, "data")` |
| `SEQUENCE_LENGTHS` | Max token lengths for ablation (L=128, 256, 512) | `[128, 256, 512]` |
| `VOCAB_MIN_FREQ` | Min token frequency to keep in vocab; rest → &lt;UNK&gt; (Phase 2 step 2) | `2` |
| `EMBEDDING_DIM` | Embedding dimension (Phase 2 step 3; must match GloVe if used) | `300` |
| `GLOVE_PATH` | Path to GloVe file (e.g. `glove.6B.300d.txt`) or `None` for random init | `None` |

---

## Running the Code

**All commands are run from the project root** (the folder containing `run_phase1.py`).

### Phase 1 – Full pipeline (recommended first run)

Runs steps 1–8 in order and then prints a short summary of outputs:

```bash
python run_phase1.py
```

### Phase 1 – Single steps

Run one step at a time (in order 1 → 2 → … → 8):

```bash
python src/phase1/step1_clean_liar.py
python src/phase1/step2_normalize_text.py
python src/phase1/step3_tokenize.py
python src/phase1/step4_sequence_length.py
python src/phase1/step5_class_balance.py
python src/phase1/step6_concept_mapping.py
python src/phase1/step7_dataset_split.py
python src/phase1/step8_evaluation_variants.py
```

### Phase 2 – Representation and encoding

Run after Phase 1 (at least through step 2). Step 2 needs step 1’s output.

```bash
python src/phase2/step1_retokenize.py
python src/phase2/step2_vocabulary_encoding.py
python src/phase2/step3_embeddings.py
```

### View Phase 1 results (without re-running)

Lists files in `data/` and shows short previews of key CSVs:

```bash
python show_phase1_results.py
```

---

## Phase 1 – Step Details

| Step | Script | Input | Output | What it does |
|------|--------|--------|--------|----------------|
| 1 | `phase1/step1_clean_liar.py` | Raw `train.tsv`, `valid.tsv`, `test.tsv` from `LIAR_DATASET_DIR` | `liar_cleaned_step1.csv` | Load Liar TSVs, select statement + label, drop missing/duplicates, clean text (HTML, URLs→&lt;URL&gt;, numbers→&lt;NUM&gt;, ALLCAPS marker, lowercase). |
| 2 | `phase1/step2_normalize_text.py` | `liar_cleaned_step1.csv` | `liar_normalized_step2.csv` | Unicode normalization, contractions, whitespace and punctuation normalization. |
| 3 | `phase1/step3_tokenize.py` | `liar_normalized_step2.csv` | `liar_tokenized_step3.csv` | Word-level tokenization (SpaCy), keep punctuation and negation. |
| 4 | `phase1/step4_sequence_length.py` | `liar_tokenized_step3.csv` | `liar_truncated_L128.csv`, `_L256.csv`, `_L512.csv` | Head truncation to L=128, 256, 512 for ablation. |
| 5 | `phase1/step5_class_balance.py` | `liar_truncated_L128.csv` | `class_weights.txt` | Binary (fake/real) distribution, class weights for weighted loss (no oversampling). |
| 6 | `phase1/step6_concept_mapping.py` | `liar_truncated_L{L}.csv` per L | `liar_concepts_step6_L128.csv`, `_L256.csv`, `_L512.csv` | Lexicon-based concept tags: Emotion, Modality, Negation (no labels used). |
| 7 | `phase1/step7_dataset_split.py` | `liar_concepts_step6_L{L}.csv` per L | `train_split_L{L}.csv`, `val_split_L{L}.csv`, `test_split_L{L}.csv` | Stratified 70–15–15 train/val/test, same `random_state=42` for all L. |
| 8 | `phase1/step8_evaluation_variants.py` | `train_split_L{L}.csv` per L | `train_eval_variants_step8_L128.csv`, `_L256.csv`, `_L512.csv` | Sufficiency (concept tokens only) and comprehensiveness (non-concept only) variants. |

---

## Phase 2 – Step Details

| Step | Script | Input | Output | What it does |
|------|--------|--------|--------|----------------|
| 1 | `phase2/step1_retokenize.py` | `liar_normalized_step2.csv` | `liar_phase2_reasoning_tokens.csv` | Retokenize for reasoning units (SpaCy; preserve punctuation, negation, special tokens). |
| 2 | `phase2/step2_vocabulary_encoding.py` | `liar_phase2_reasoning_tokens.csv` | `v`vocab_word2idx.json``, `vocab_params.txt`, `encoded_L{L}.pkl` | Vocab from train only; token→index; OOV→&lt;UNK&gt;; encode per L. |
| 3 | `phase2/step3_embeddings.py` | `vocab_word2idx.json` | `embedding_matrix.npy`, `embedding_params.txt` | GloVe (if path set) or random init; PAD=0; fine-tunable. |

---

## Embedding layer (Phase 2 Step 3) – details

Step 3 builds an **embedding matrix** of shape `(vocab_size, EMBEDDING_DIM)` for the model’s embedding layer.

### GloVe path not set (default)

- **`GLOVE_PATH = None`** in `src/config.py` → **random initialization** is used.
- All tokens (except `[PAD]`) get small random vectors; `[PAD]` stays zeros.
- This is **fine for development and testing** and is enough to run the rest of the pipeline.

### Using pretrained GloVe (optional)

For better semantic grounding (e.g. for the final model or paper):

1. **Download GloVe** (e.g. [Stanford GloVe](https://nlp.stanford.edu/projects/glove/) → GloVe 6B, then unzip and use e.g. `glove.6B.300d.txt`).
2. In **`src/config.py`** set:
   - `EMBEDDING_DIM = 300` (must match the GloVe file you use: 50, 100, 200, or 300).
   - `GLOVE_PATH = r"C:\path\to\glove.6B.300d.txt"` (or your actual path).
3. Re-run:
   ```bash
   python src/phase2/step3_embeddings.py
   ```
   You should see “Pretrained (GloVe): True” and “Vocab tokens in GloVe: …”.

### Outputs

| File | Description |
|------|-------------|
| `data/embedding_matrix.npy` | NumPy array `(vocab_size, EMBEDDING_DIM)` float32. Use to initialize an embedding layer (e.g. in PyTorch) and fine-tune during training. |
| `data/embedding_params.txt` | `vocab_size`, `embedding_dim`, `pretrained` (True/False), `glove_matched` (how many vocab tokens had a GloVe vector). |

---

## Output Files Reference

- **Single files:** `liar_cleaned_step1.csv`, `liar_normalized_step2.csv`, `liar_tokenized_step3.csv`, `class_weights.txt`, `liar_phase2_reasoning_tokens.csv`, `v`vocab_word2idx.json``, `vocab_params.txt`, `embedding_matrix.npy`, `embedding_params.txt`.
- **Per L (128, 256, 512):** `liar_truncated_L{L}.csv`, `liar_concepts_step6_L{L}.csv`, `train_split_L{L}.csv`, `val_split_L{L}.csv`, `test_split_L{L}.csv`, `train_eval_variants_step8_L{L}.csv`, `encoded_L{L}.pkl` (Phase 2: indices + labels).

All paths are under `data/` (or whatever `DATA_DIR` is set to in `config.py`).

---

## Sequence Length Variants (L = 128, 256, 512)

The pipeline produces separate truncated datasets and splits for **L = 128, 256, and 512** so you can compare:

- **Accuracy vs L** (e.g. plateau at longer L).
- **Reasoning faithfulness vs L** (e.g. peak at moderate L).

Train/val/test use the same row assignment across L (same random seed and stratification).

---

## Data Files (Raw Input)

- **Raw Liar:** `train.tsv`, `valid.tsv`, `test.tsv` in the directory set by `LIAR_DATASET_DIR` in `src/config.py`.
- **Processed:** All other outputs live in `data/` as listed above.

---

## License and Citation

Use and cite according to your institution’s and the Liar dataset’s terms.
