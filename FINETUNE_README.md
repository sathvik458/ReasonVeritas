# Fine-tune branch — quick start (M3 Pro 36 GB)

This branch (`frozen-bert`) now ships **two parallel pipelines**:

| Pipeline | Purpose | LIAR macro-F1 (seed 42) |
|---|---|---|
| **Frozen** (`train_model.py`, `train_phase4.py`) | Cached DistilBERT features + light head. Cheap baseline, runs on 8 GB. | Phase 3: 0.633, Phase 4: 0.631 |
| **Fine-tuned** (`train_finetune.py`, `train_phase4_finetune.py`) | End-to-end, top-N DistilBERT layers trainable. Needs ≥16 GB. | *expected ~0.66-0.69 with `--unfreeze 2`* |

The frozen pipeline is unchanged — both are kept for clean ablations in the paper.

---

## Setup

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Make sure the cached files from previous runs exist in `data/`:

```
ls data/{train,val,test}_split_L128.csv          # Phase 1 step7 output
ls data/bert_features_{train,val,test}_L128.npz  # Phase 2 step3 (only needed for Phase 4 mask alignment)
ls data/concept_masks_{train,val,test}_L128.npz  # Phase 4 mask cache (auto-built by train_phase4.py)
```

If any are missing, re-run Phase 1 and Phase 2 in order:

```
python src/phase1/step1_clean_liar.py
python src/phase1/step2_normalize_text.py
python src/phase1/step3_tokenize.py
python src/phase1/step4_sequence_length.py
python src/phase1/step5_class_balance.py
python src/phase1/step6_concept_mapping.py
python src/phase1/step7_dataset_split.py
python src/phase2/step3_embeddings.py            # caches BERT features + subword_to_word
```

---

## Run on M3 Pro (recommended order)

### 1. Sanity check the frozen baseline (5 min)
```
python src/phase3/train_model.py --L 128
python src/phase4/train_phase4.py  --L 128
```
Should reproduce ~0.633 / 0.631 macro-F1. The frozen Phase 4 trainer now uses
the **gated aux loss** (each concept head's auxiliary classifier only trains
on examples containing that concept) — so the dead-emotion-head problem is fixed
in this run too. Expect E_mean to climb from ~0.04 to ~0.20+.

### 2. Fine-tuned Phase 3 (~30-45 min on M3 Pro)
```
python src/phase3/train_finetune.py --L 128 --unfreeze 2
```
Knobs:
- `--unfreeze 2`  unfreezes top 2 DistilBERT transformer blocks (default; safest)
- `--unfreeze 4`  unfreezes top 4 blocks (more capacity, higher overfit risk)
- `--unfreeze 6`  unfreezes the entire transformer (full fine-tune)
- `--bert_lr 2e-5` `--head_lr 5e-4` (defaults; ULMFiT-style two-group LR)
- `--batch 16 --accum 2` effective batch 32; bump `--batch 32 --accum 1` if memory allows.

Starts from `distilbert-base-uncased` weights, trains end-to-end. Best
checkpoint saved to `models/phase3_finetune_L128_uf2_best.pt`. Logs in
`logs/phase3_finetune_L128_uf2.csv`. Final JSON in
`results/phase3_finetune_L128_uf2_seed42.json`.

### 3. Fine-tuned Phase 4 (~45-60 min on M3 Pro)
```
python src/phase4/train_phase4_finetune.py --L 128 --unfreeze 2
```
Same knobs as Phase 3. **New in this script vs. the frozen version**:
1. End-to-end DistilBERT fine-tuning.
2. Concept-presence-gated aux loss (the dead-head fix).
3. **Attention-grounded rationales** — top-K subword tokens per concept head,
   decoded via the tokenizer. Sample dump at:
   `logs/rationales_phase4_finetune/sample_phase4_finetune_L128_uf2.txt`

The rationale dump is what makes the paper's "faithful reasoning" claim
substantiable: each step cites the actual tokens the head attended to with
their attention weights, not just a templated score-to-text mapping.

### 4. Multi-seed (the number you'll report in the paper)
After confirming the fine-tune runs work end-to-end, run 3 seeds and average:
```
for s in 41 42 43; do
  python src/phase3/train_finetune.py --L 128 --unfreeze 2 --seed $s
  python src/phase4/train_phase4_finetune.py --L 128 --unfreeze 2 --seed $s
done
```
Check `results/*_finetune_*.json`, take the mean ± std of `test_macro_f1`.

---

## What changed on this branch

New files:
- `src/meta_encoder.py` — shared LIAR metadata encoder (party + credit + speaker/subject/context embeddings)
- `src/phase3/bert_finetune.py` — `FineTunedBERTClassifier` model
- `src/phase3/train_finetune.py` — Phase 3 fine-tune trainer
- `src/phase4/cot_finetune.py` — `CoTModelFineTune` model
- `src/phase4/train_phase4_finetune.py` — Phase 4 fine-tune trainer with gated aux loss + attention-grounded rationales

Edited files:
- `src/phase3/bilstm_attention.py` — accepts `meta_vocabs` (uses MetaEncoder if set)
- `src/phase3/train_model.py` — uses `meta_encoder` module instead of inline metadata logic
- `src/phase4/cot_model.py` — accepts `meta_vocabs`
- `src/phase4/train_phase4.py` — uses `gated_aux_loss` (dead-head fix), drops the inline metadata logic
- `src/phase4/rationale.py` — adds `generate_rationale_with_tokens()` while keeping the score-only function for backward compat

---

## Memory notes

DistilBERT (66M params) + 16 batch + L=128 + activations for backprop ≈ 4-6 GB
on MPS. With `--unfreeze 2` you're storing gradients for ~14M BERT params on
top, so plan on 8-10 GB peak. Should fit comfortably on 36 GB.

If you see OOM, drop `--batch 8 --accum 4` (effective batch stays at 32).
