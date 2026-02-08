# Project Title: Cater-polar

## Overview
Cater-polar is a project designed to process and clean datasets related to the Liar dataset. The primary goal is to prepare the data for further analysis or machine learning tasks.

## Directory Structure
```
Cater-polar/
	README.md
data/
	liar_cleaned_step1.csv
	README
	test.tsv
	train.tsv
	valid.tsv
src/
	step1_clean_liar.py
```

## Installation
To set up the project, ensure you have Python installed on your machine. You can download it from [python.org](https://www.python.org/downloads/).

### Required Libraries
This project requires the following Python libraries:
- `pandas`: For data manipulation and analysis.

You can install the required libraries using pip:
```bash
pip install pandas numpy scikit-learn
```

## Usage
To run the data cleaning script, execute the following command in your terminal:
```bash
python src/step1_clean_liar.py
```

## Data Files
- `liar_cleaned_step1.csv`: The cleaned dataset after processing.
- `train.tsv`, `valid.tsv`, `test.tsv`: Datasets for training, validation, and testing.

## Processing Steps (Detailed)
This project processes the Liar dataset through an 8-step pipeline. Each step is implemented as a script in the `src/` directory; the scripts are intended to be run in sequence from `step1` → `step8`.

- **Step 1 — Cleaning (`src/step1_clean_liar.py`):**
	- **Purpose:** Load the raw Liar files and perform baseline cleaning to produce a consistent CSV dataset.
	- **Input:** original `.tsv` files (e.g., `data/train.tsv`, `data/valid.tsv`, `data/test.tsv`).
	- **Output:** `data/liar_cleaned_step1.csv` containing normalized columns, removed empty or malformed rows, and basic sanitization (trim whitespace, remove stray control characters).
	- **Key actions:** column selection/renaming, drop/repair missing values, simple text sanitization.

- **Step 2 — Normalization (`src/step2_normalize_text.py`):**
	- **Purpose:** Apply text normalization to the cleaned statements so downstream tokenizers and models see consistent input.
	- **Input:** `data/liar_cleaned_step1.csv`.
	- **Output:** `data/liar_normalized_step2.csv` with normalized text (lowercasing, unicode normalization, URL/email removal or masking, normalized punctuation and whitespace).
	- **Key actions:** standardize case, normalize unicode, remove/replace URLs and emails, collapse repeated spaces, optionally strip non-printable characters.

- **Step 3 — Tokenization (`src/step3_tokenize.py`):**
	- **Purpose:** Convert normalized text into tokens suitable for modelling or length analysis.
	- **Input:** `data/liar_normalized_step2.csv`.
	- **Output:** `data/liar_tokenized_step3.csv` which includes tokenized versions of the statements and optionally token counts or token sequences depending on configuration.
	- **Key actions:** apply a tokenizer (e.g., simple whitespace, `nltk`, or a subword tokenizer), save tokens and token counts for each record.

- **Step 4 — Sequence Length & Truncation (`src/step4_sequence_length.py`):**
	- **Purpose:** Analyze token length distribution and produce truncated variants to fixed maximum lengths used for model experiments.
	- **Input:** `data/liar_tokenized_step3.csv`.
	- **Output:** truncated datasets such as `data/liar_truncated_L128.csv` and `data/liar_truncated_L32.csv` (examples in the `data/` folder), plus a small length-distribution report.
	- **Key actions:** compute length histograms, choose sensible cutoff(s), truncate/pad token sequences and store the resulting datasets.

- **Step 5 — Class Balance (`src/step5_class_balance.py`):**
	- **Purpose:** Inspect class distribution, compute class weights for training, and optionally generate balanced sampling variants.
	- **Input:** cleaned/normalized/truncated datasets (e.g., outputs from earlier steps).
	- **Output:** `data/class_weights.txt` containing computed weights and `data/train_eval_variants_step8.csv` (or other balanced variants) used for model training/validation.
	- **Key actions:** compute per-class counts, derive inverse-frequency weights, save sampling variants or instructions for weighted training.

- **Step 6 — Concept Mapping (`src/step6_concept_mapping.py`):**
	- **Purpose:** Map statements to higher-level concepts or categories to support concept-aware experiments or feature extraction.
	- **Input:** cleaned/normalized data (from previous steps) and any mapping resources or heuristics.
	- **Output:** `data/liar_concepts_step6.csv` which augments records with concept labels or flags.
	- **Key actions:** apply mapping rules, keyword heuristics, or lookup tables; attach concept metadata to each row.

- **Step 7 — Dataset Split (`src/step7_dataset_split.py`):**
	- **Purpose:** Create train/validation/test splits (and any cross-validation folds) used for experiments.
	- **Input:** processed dataset (finalized after previous steps).
	- **Output:** split files such as `data/train_split.csv`, `data/val_split.csv`, `data/test_split.csv` and TSV variants used by other tooling (`train.tsv`, `valid.tsv`, `test.tsv`).
	- **Key actions:** stratified splitting to preserve class ratios, optional random seed control, and saving both CSV and TSV variants.

- **Step 8 — Evaluation Variants (`src/step8_evaluation_variants.py`):**
	- **Purpose:** Generate evaluation set variants and helper outputs for downstream evaluation and ablation studies.
	- **Input:** training and test splits, plus concept labels or truncated variants as needed.
	- **Output:** `data/train_eval_variants_step8.csv` and any prepared evaluation TSVs; scripts may also compute baseline metrics or helper files for model evaluation.
	- **Key actions:** produce variant datasets (e.g., different truncation/class-balance combinations), prepare evaluation-ready TSVs, and optionally compute quick baseline metrics.




