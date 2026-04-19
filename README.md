# Cater-polar / ReasonVeritas

**Confidence-Calibrated Chain-of-Thought Reasoning for Interpretable and Reliable Deep Learning Models**
*(LIAR dataset for fake-news / claim prediction)*

---

## Overview

ReasonVeritas is a four-phase pipeline for interpretable, confidence-aware fake-news classification on the LIAR dataset. The pipeline cleans and tokenizes the corpus, encodes it with DistilBERT, runs a strong BiLSTM-Attention baseline, and finally adds a Chain-of-Thought (CoT) reasoning head with three concept-gated attention channels (Emotion, Modality, Negation) and faithful, attention-grounded rationales.

| Phase | Purpose |
|---|---|
| **Phase 1** | Clean → normalize → tokenize → truncate (L=128) → class balance → concept mapping → stratified splits → eval variants. |
| **Phase 2** | Reasoning-unit retokenization (legacy) → vocabulary encoding (legacy) → **DistilBERT contextual feature caching + subword↔word alignment** (current). |
| **Phase 3** | BiLSTM + multi-head self-attention classifier with optional metadata (party + credit-history + speaker/subject/context embeddings). Two variants: **frozen** features and **end-to-end fine-tuned** DistilBERT. |
| **Phase 4** | Chain-of-Thought head: 3 concept-gated attention heads + auxiliary multi-task losses + attention-grounded rationales. Same two variants (frozen / fine-tuned). |

All pipeline outputs are written to `data/`. Paths and constants are centralized in `src/config.py`.

---

## What's changed since the previous README

The earlier README only documented Phase 1 and Phase 2. This version adds Phase 3 and Phase 4 plus several substantial changes to the upstream phases:

- **DistilBERT replaces GloVe.** Phase 2 step 3 no longer builds a static GloVe-or-random embedding matrix. It now caches `last_hidden_state` from `distilbert-base-uncased` and a `subword_to_word` alignment used by Phase 4's concept masks. The `EMBEDDING_DIM` / `GLOVE_PATH` config entries are no longer used by Phase 3 or Phase 4.
- **Sequence length is standardized at L=128.** L=256 and L=512 still build, but all reported numbers and run commands use L=128.
- **Metadata encoder.** `src/meta_encoder.py` adds party (one-hot), credit-history counts (5 numeric, scaled), and learnable embeddings for speaker / subject / context (Karimi & Tang 2019). Wired into both Phase 3 and Phase 4 backbones.
- **Phase 3 added.** BiLSTM + 4-head self-attention + mean+max pooling classifier. See "Phase 3" below for the frozen and fine-tuned variants.
- **Phase 4 added.** Same backbone with three concept-gated attention heads (Emotion / Modality+scope / Negation+scope), auxiliary multi-task losses, and template + attention-grounded rationales.
- **Gated auxiliary loss (dead-head fix).** Each concept head's auxiliary classifier is now trained only on examples where the corresponding concept is actually present in the input. This fixed the "dead emotion head" problem (`E_mean ≈ 0.04`) we saw with flat aux loss.
- **Attention-grounded rationales.** `src/phase4/rationale.py` now also emits the top-K subword tokens each concept head attended to, decoded via the tokenizer. Sample dump in `logs/rationales_phase4*/`.
- **End-to-end fine-tune branch.** `train_finetune.py` (Phase 3) and `train_phase4_finetune.py` (Phase 4) unfreeze the top-N DistilBERT transformer blocks for full end-to-end training (ULMFiT-style two-group LR). See `FINETUNE_README.md` for the M3 Pro quick start.

---

## Directory Structure

```
Reasonveritas/
├── README.md
├── FINETUNE_README.md         # Quick start for the fine-tuned pipeline (≥16 GB GPU/MPS)
├── run_phase1.py              # Run all Phase 1 steps and show summary
├── show_phase1_results.py     # View Phase 1 outputs without re-running
├── requirements.txt
├── data/                      # All pipeline outputs (created on first run)
│   ├── liar_cleaned_step1.csv
│   ├── liar_normalized_step2.csv
│   ├── liar_tokenized_step3.csv
│   ├── liar_truncated_L128.csv (also _L256, _L512)
│   ├── class_weights.txt
│   ├── liar_concepts_step6_L128.csv (also _L256, _L512)
│   ├── train_split_L128.csv, val_split_L128.csv, test_split_L128.csv
│   ├── train_eval_variants_step8_L128.csv
│   ├── bert_features_{train,val,test}_L128.npz   # Phase 2 step 3 (DistilBERT)
│   ├── subword_to_word_{train,val,test}_L128.npz # Phase 2 step 3 (alignment)
│   └── concept_masks_{train,val,test}_L128.npz   # built lazily by Phase 4
├── models/                    # Saved checkpoints (best val macro-F1)
│   ├── phase3_L128_best.pt, phase3_finetune_L128_uf2_best.pt
│   └── phase4_L128_best.pt,  phase4_finetune_L128_uf2_best.pt
├── results/                   # Final test JSONs (one per run)
├── logs/                      # CSV training logs + rationale dumps
└── src/
    ├── config.py              # Paths, BERT_MODEL_NAME, sequence lengths
    ├── meta_encoder.py        # Party + credit + speaker/subject/context embeddings
    ├── utils_logger.py
    ├── phase1/                # Dataset + preprocessing (steps 1–8)
    ├── phase2/                # Reasoning tokens + vocab (legacy) + BERT features (current)
    │   ├── step1_retokenize.py
    │   ├── step2_vocabulary_encoding.py
    │   └── step3_embeddings.py        # DistilBERT contextual features + subword alignment
    ├── phase3/                # BiLSTM-Attention classifier
    │   ├── bilstm_attention.py        # backbone (frozen-feature variant)
    │   ├── train_model.py             # frozen-feature trainer
    │   ├── bert_finetune.py           # FineTunedBERTClassifier (end-to-end DistilBERT)
    │   └── train_finetune.py          # fine-tune trainer
    └── phase4/                # Chain-of-Thought reasoning head
        ├── cot_model.py               # 3 concept-gated heads + fusion (frozen variant)
        ├── train_phase4.py            # frozen-feature trainer (gated aux loss)
        ├── cot_finetune.py            # CoTModelFineTune (end-to-end DistilBERT)
        ├── train_phase4_finetune.py   # fine-tune trainer + attention rationales
        └── rationale.py               # template + attention-grounded rationale generators
```

---

## Prerequisites

- **Python** 3.8+
- **LIAR dataset:** place `train.tsv`, `valid.tsv`, `test.tsv` in a folder and set that path in `src/config.py` (`LIAR_DATASET_DIR`).
- **PyTorch** with MPS (Apple Silicon) or CUDA. Frozen pipeline runs comfortably on 8 GB RAM; fine-tuned pipeline needs ≥16 GB.

---

## Installation

From the project root:

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

**Main dependencies:** `pandas`, `numpy`, `scikit-learn`, `spacy`, `emoji`, `beautifulsoup4`, `contractions`, `torch`, `transformers`.

---

## Configuration

Edit **`src/config.py`** before the first run:

| Variable | Meaning | Default |
|----------|---------|---------|
| `LIAR_DATASET_DIR` | Folder containing raw `train.tsv`, `valid.tsv`, `test.tsv` | (set me) |
| `DATA_DIR` | Where pipeline outputs are saved | `<repo>/data` |
| `BERT_MODEL_NAME` | HuggingFace model id used by Phase 2 step 3 + Phase 3/4 fine-tune | `distilbert-base-uncased` |
| `SEQUENCE_LENGTHS` | Truncation lengths produced by Phase 1 | `[128, 256, 512]` |

> The legacy `EMBEDDING_DIM` and `GLOVE_PATH` fields are still in `config.py` but are no longer consumed by Phase 3 or Phase 4. They only affect the legacy Phase 2 step 2 vocabulary encoding, which downstream code does not import.

---

## Running the full pipeline

All commands run from the project root. The recommended order on a fresh checkout is:

```bash
# Phase 1 — preprocess + splits
python src/phase1/step1_clean_liar.py
python src/phase1/step2_normalize_text.py
python src/phase1/step3_tokenize.py
python src/phase1/step4_sequence_length.py
python src/phase1/step5_class_balance.py
python src/phase1/step6_concept_mapping.py
python src/phase1/step7_dataset_split.py
python src/phase1/step8_evaluation_variants.py        # optional, for sufficiency/comprehensiveness eval

# Phase 2 — DistilBERT contextual features + subword alignment (only step 3 is needed by Phase 3/4)
python src/phase2/step3_embeddings.py

# Phase 3 — BiLSTM-Attention classifier (frozen features)
python src/phase3/train_model.py --L 128

# Phase 4 — Chain-of-Thought reasoning head (frozen features)
python src/phase4/train_phase4.py  --L 128
```

Or alternatively, run all of Phase 1 in one shot:

```bash
python run_phase1.py
```

For end-to-end DistilBERT fine-tuning (≥16 GB), see **Phase 3 — Fine-tuned variant** and **Phase 4 — Fine-tuned variant** below, and the `FINETUNE_README.md` quick start.

---

## Phase 1 — Step Details

| Step | Script | Output | What it does |
|------|--------|--------|--------------|
| 1 | `phase1/step1_clean_liar.py` | `liar_cleaned_step1.csv` | Load LIAR TSVs, select statement + label, drop missing/duplicates, clean text (HTML, URLs→`<URL>`, numbers→`<NUM>`, ALLCAPS marker, lowercase). |
| 2 | `phase1/step2_normalize_text.py` | `liar_normalized_step2.csv` | Unicode normalization, contractions, whitespace and punctuation normalization. |
| 3 | `phase1/step3_tokenize.py` | `liar_tokenized_step3.csv` | Word-level tokenization (SpaCy), keep punctuation and negation. |
| 4 | `phase1/step4_sequence_length.py` | `liar_truncated_L128.csv` (+ L256, L512) | Head truncation for ablation. All current results use L=128. |
| 5 | `phase1/step5_class_balance.py` | `class_weights.txt` | Binary (fake/real) distribution + class weights for weighted loss. |
| 6 | `phase1/step6_concept_mapping.py` | `liar_concepts_step6_L{L}.csv` | Lexicon-based concept tags: Emotion, Modality, Negation. Ground truth for Phase 4's concept heads — no labels are used here. |
| 7 | `phase1/step7_dataset_split.py` | `train/val/test_split_L{L}.csv` | Stratified 70/15/15 with `random_state=42`. Same row assignment across L. |
| 8 | `phase1/step8_evaluation_variants.py` | `train_eval_variants_step8_L{L}.csv` | Sufficiency (concept tokens only) and comprehensiveness (non-concept only) variants for faithfulness eval. |

---

## Phase 2 — Step Details

Only **step 3** is consumed by Phase 3 and Phase 4. Steps 1 and 2 remain for the legacy GloVe-style pipeline and ablation purposes.

| Step | Script | Output | What it does |
|------|--------|--------|--------------|
| 1 | `phase2/step1_retokenize.py` | `liar_phase2_reasoning_tokens.csv` | (Legacy) Retokenize for reasoning units. Not required by Phase 3/4. |
| 2 | `phase2/step2_vocabulary_encoding.py` | `vocab_word2idx.json`, `encoded_L{L}.pkl` | (Legacy) Build vocabulary + integer-encode. Not required by Phase 3/4. |
| 3 | `phase2/step3_embeddings.py` | `bert_features_{train,val,test}_L128.npz`, `subword_to_word_{train,val,test}_L128.npz` | **Cache `distilbert-base-uncased` `last_hidden_state` (B, T=128, 768) + a per-token mapping from WordPiece subwords back to whitespace word indices**, used by Phase 4 to align concept masks to subwords. |

---

## Phase 3 — BiLSTM-Attention Classifier

Phase 3 trains the strong baseline: DistilBERT features → BiLSTM → multi-head self-attention → mean+max pooling → MLP head, with optional metadata (party + credit history + speaker/subject/context embeddings) concatenated into the fused vector before the classifier.

**Backbone (`src/phase3/bilstm_attention.py`)**

- BiLSTM over BERT `last_hidden_state`, `hidden_size=192`, single layer.
- 4-head self-attention over the BiLSTM outputs (with the attention mask from the tokenizer).
- Mean + max pool of attended outputs → fused 768-dim representation.
- Optional metadata branch:
  - `meta_dim` linear path (legacy, `meta_dim=15` = party-7 + credit-5 + presence flags-3), or
  - `meta_vocabs` path → `MetaEncoder` (party-7 + credit-5 + speaker-32 + subject-16 + context-16 = 96-dim out).
- Final classifier: `[fused | meta] → Linear → GELU → Dropout → Linear → 2 logits`.
- Dropout: `0.2` on attention outputs and inside the classifier MLP.

**Training (`src/phase3/train_model.py`, frozen features)**

- Reads `bert_features_*_L128.npz` from disk; BERT is *not* loaded at train time.
- Smoothed cross-entropy (`smoothing=0.05`) with class weights from `class_weights.txt`.
- AdamW, single LR group `5e-4`, cosine schedule.
- Batch 32, ~10 epochs with early stop on val macro-F1.
- Best checkpoint → `models/phase3_L128_best.pt`. Test JSON → `results/phase3_L128_seed42.json`.

**Run (frozen baseline, ~5 min on M2 Air)**

```bash
python src/phase3/train_model.py --L 128
```

Reference number: **macro-F1 ≈ 0.633** on the LIAR test split (seed 42).

### Phase 3 — Fine-tuned variant (end-to-end DistilBERT)

`src/phase3/bert_finetune.py` adds `FineTunedBERTClassifier`, which loads `distilbert-base-uncased` live, freezes its embeddings, unfreezes the top-N transformer blocks, and feeds `last_hidden_state` into the same BiLSTM-Attention head. `src/phase3/train_finetune.py` is the matching trainer:

- On-the-fly tokenization with `DistilBertTokenizer` (no NPZ cache reads).
- Two-group AdamW (ULMFiT-style): `bert_lr=2e-5`, `head_lr=5e-4`.
- Linear warmup (10%) + linear decay floored at 0.1.
- Gradient accumulation: defaults `--batch 16 --accum 2` → effective batch 32. Drop to `--batch 8 --accum 4` if you OOM.

**Run (~30–45 min on M3 Pro 36 GB)**

```bash
python src/phase3/train_finetune.py --L 128 --unfreeze 2
```

Knobs: `--unfreeze {2,4,6}` (top-N transformer blocks; 6 = full transformer), `--bert_lr`, `--head_lr`, `--batch`, `--accum`, `--epochs`, `--dropout`, `--seed`. Best checkpoint → `models/phase3_finetune_L128_uf2_best.pt`. Expected: **macro-F1 ≈ 0.66–0.69** with `--unfreeze 2`.

---

## Phase 4 — Chain-of-Thought Reasoning Head

Phase 4 adds three **concept-gated attention heads** on top of the Phase 3 backbone, each looking at a single reasoning channel of the input:

1. **Emotion** — affective lexicon hits (e.g. `terrible`, `outrageous`).
2. **Modality + scope** — modal verbs and their scope (e.g. `might`, `should`, `claimed that ...`).
3. **Negation + scope** — negation cues and their scope (e.g. `not`, `never`).

The masks come from Phase 1 step 6's lexicon-based concept mapping and are aligned to DistilBERT subwords via the cached `subword_to_word` map (Phase 2 step 3).

**Architecture (`src/phase4/cot_model.py`)**

- Shared BiLSTM + self-attention base (same as Phase 3).
- Three `ConceptHead` modules: each is a small attention pooler restricted to its concept mask, producing a per-concept score plus an auxiliary 2-class logit.
- Fusion: `[base_pool | em_pool | mo_pool | ne_pool | meta] → Linear → GELU → Dropout → Linear → 2 logits`.
- Returns an 11-tuple: main logits, three concept scores, three aux logits, three attention distributions, plus the base attention.

**Training (`src/phase4/train_phase4.py`, frozen features)**

- Main loss: smoothed CE with class weights.
- **Gated auxiliary loss (the dead-head fix):** each concept's aux classifier is only trained on examples where that concept actually appears in the input. Without this, the emotion head was learning nothing (`E_mean ≈ 0.04`); with it we get `E_mean ≥ 0.20`.
- Coverage penalty (target ≈ 0.3) on each concept attention to prevent the head from collapsing onto a single token.
- Per-step rationale generation via `rationale.generate_rationale` (template-based: maps scores to a fixed reasoning sentence).
- Best checkpoint → `models/phase4_L128_best.pt`. Test JSON → `results/phase4_L128_seed42.json`.

**Run (frozen baseline, ~5 min on M2 Air)**

```bash
python src/phase4/train_phase4.py --L 128
```

Reference number: **macro-F1 ≈ 0.631** on the LIAR test split (seed 42).

### Phase 4 — Fine-tuned variant (end-to-end DistilBERT + attention-grounded rationales)

`src/phase4/cot_finetune.py` wraps a live DistilBERT around the existing `CoTModel` so the head architecture is identical to the frozen variant — clean ablation. `src/phase4/train_phase4_finetune.py` is the trainer. **What's new compared to the frozen Phase 4 trainer:**

1. End-to-end DistilBERT fine-tuning (selectively unfreeze top-N blocks).
2. Concept-presence-gated aux loss (the dead-head fix, same as frozen).
3. **Attention-grounded rationales.** `rationale.generate_rationale_with_tokens` decodes the top-K subword tokens each concept head attended to (WordPiece `##` continuations stripped, special tokens filtered), with their attention weights. Sample dump → `logs/rationales_phase4_finetune/sample_phase4_finetune_L128_uf2.txt`.

The attention-grounded rationale is what makes the paper's "faithful reasoning" claim substantiable: each step cites the actual tokens the head attended to with their attention weights, not just a templated score-to-text mapping.

**Run (~45–60 min on M3 Pro 36 GB)**

```bash
python src/phase4/train_phase4_finetune.py --L 128 --unfreeze 2
```

Knobs: same as Phase 3 fine-tune. Best checkpoint → `models/phase4_finetune_L128_uf2_best.pt`. Test JSON → `results/phase4_finetune_L128_uf2_seed42.json`.

### Multi-seed (paper numbers)

After confirming end-to-end runs work, average over three seeds:

```bash
for s in 41 42 43; do
  python src/phase3/train_finetune.py        --L 128 --unfreeze 2 --seed $s
  python src/phase4/train_phase4_finetune.py --L 128 --unfreeze 2 --seed $s
done
```

Take mean ± std of `test_macro_f1` across the three `results/*_finetune_*_seed{41,42,43}.json` files.

---

## Metadata Encoder (`src/meta_encoder.py`)

Both Phase 3 and Phase 4 backbones accept an optional `meta_vocabs` dict that switches the metadata branch from the legacy 15-dim linear projection to a `MetaEncoder` module, motivated by Karimi & Tang (2019) (speaker history is the single most predictive feature on LIAR):

- **Party** — 7-way one-hot (dem, rep, ind, libertarian, green, none, other).
- **Credit history** — 5 numeric counts (barely_true, false, half_true, mostly_true, pants_on_fire), MinMax-scaled on train.
- **Speaker / subject / context** — top-K (default 500/100/80) vocabularies built on train, with `<UNK>` at index 0; learnable `nn.Embedding`s of size 32 / 16 / 16.
- Combined output: 96-dim, concatenated into the fused vector before the classifier.

`fit_metadata(train_df)` returns the vocab + scaler; `encode_metadata(df, vocabs, scaler)` produces the per-row tensors used by the trainers.

---

## Output Files Reference

- **Phase 1:** `liar_cleaned_step1.csv`, `liar_normalized_step2.csv`, `liar_tokenized_step3.csv`, `liar_truncated_L{L}.csv`, `class_weights.txt`, `liar_concepts_step6_L{L}.csv`, `train/val/test_split_L{L}.csv`, `train_eval_variants_step8_L{L}.csv`.
- **Phase 2 (current):** `bert_features_{train,val,test}_L128.npz`, `subword_to_word_{train,val,test}_L128.npz`.
- **Phase 4 mask cache:** `concept_masks_{train,val,test}_L128.npz` (built lazily on first run).
- **Models:** `models/phase{3,4}_L128_best.pt`, `models/phase{3,4}_finetune_L128_uf{N}_best.pt`.
- **Results:** `results/phase{3,4}_L128_seed{N}.json`, `results/phase{3,4}_finetune_L128_uf{N}_seed{N}.json`.
- **Logs:** `logs/phase{3,4}_*.csv` for training metrics; `logs/rationales_phase4*/` for rationale dumps.

All paths are under `data/` / `models/` / `results/` / `logs/` (or whatever you set in `config.py`).

---

## Sequence Length Variants

The pipeline produces L=128, 256, and 512 truncated splits, but **all reported numbers and recommended commands use L=128.** The longer variants are kept for ablations on accuracy vs. L and reasoning faithfulness vs. L.

---

## License and Citation

Use and cite according to your institution's and the LIAR dataset's terms.

Key references:
- Wang, W. Y. (2017). *"Liar, Liar Pants on Fire": A New Benchmark Dataset for Fake News Detection.* ACL.
- Karimi, H. & Tang, J. (2019). *Learning Hierarchical Discourse-level Structure for Fake News Detection.* NAACL.
- Howard, J. & Ruder, S. (2018). *Universal Language Model Fine-tuning for Text Classification.* ACL. (ULMFiT-style two-group LR.)
- Sun, C. et al. (2019). *How to Fine-Tune BERT for Text Classification?* CCL.
- Sanh, V. et al. (2019). *DistilBERT, a distilled version of BERT.* NeurIPS EMC^2 Workshop.
