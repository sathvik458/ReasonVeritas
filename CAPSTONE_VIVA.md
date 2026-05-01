# Capstone viva crib sheet — ReasonVeritas

Print or open this on a second screen during the panel. Six sections, each ≤ 1 minute to walk through.

## 1 · Headline numbers

| Setup | Aggregate macro-F1 | LIAR slice | CoAID slice |
|---|---|---|---|
| Phase 3 (BiLSTM + Attention) — LIAR-only baseline | **0.633** | 0.633 | n/a |
| Phase 4 (CoT, frozen) — LIAR-only | **0.631** | 0.631 | n/a |
| Phase 4 (CoT, frozen) — **joint LIAR + CoAID, balanced** | **0.716** | 0.631 | **0.952** |
| Phase 4 (CoT, fine-tuned, top-2 layers) — joint | (your fine-tune number) | | |

Headline takeaway in 15 seconds: *"Adding the CoT concept-gated heads matches the BiLSTM+attention baseline on LIAR while making the model's reasoning interpretable. Joint training with CoAID under a domain-balanced sampler then lifts aggregate macro-F1 by ~9 points and pushes CoAID accuracy to 0.95."*

## 2 · What's novel (the elevator pitch)

Three contributions:

1. **Concept-gated CoT reasoning head.** Three parallel attention channels — emotion, modality+scope, negation+scope — each restricted to its own lexicon-tagged tokens. Each channel produces a per-claim score and top-attended tokens, used both in the classifier and in the human-readable rationale.
2. **Cross-domain joint training.** A `has_meta` per-row mask lets the metadata branch (party, credit history, speaker/subject/context embeddings) coexist with metadata-less corpora (CoAID). One model trains on both political and health misinformation without contaminating either branch.
3. **Domain-balanced sampling.** A `WeightedRandomSampler` over `(dataset_source, binary_label)` buckets prevents the smaller corpus (CoAID) from being drowned in batches dominated by LIAR.

## 3 · Live demo flow (≤ 90 seconds)

1. Open `logs/rationales_phase4/curated_for_viva.txt` (you generate this with `python src/phase4/curate_rationales.py --tag phase4_merged_L128`).
2. Pick **2 LIAR examples** (one fake, one real) and **1 CoAID example**. Walk through:
   - Read the claim aloud.
   - Read the three concept channel scores.
   - Point at one or two top-attended tokens per channel and explain why they make sense.
   - Read the conclusion line — model verdict + confidence.
3. Close with: *"This is what we mean by interpretable — every prediction has a per-step rationale traceable to actual tokens the model attended to."*

## 4 · Likely panel questions and short answers

**Q: Why DistilBERT instead of BERT or RoBERTa?**
A: Same architecture family, ~half the parameters (66M vs 110M), runs on 8 GB unified memory with no quantization. Empirically within 1 point of BERT-base on classification. Lets us train end-to-end on a laptop.

**Q: Why concept-gated attention instead of just regular attention?**
A: Regular self-attention can attend anywhere, which makes "rationales" arbitrary. Restricting each concept head to its own lexicon-tagged tokens forces the rationale to be in the right semantic category, so the resulting CoT trace is structured by linguistic theory (emotion / modality / negation) rather than by model whim.

**Q: Why does the CoT model get the same accuracy as the non-CoT baseline?**
A: That's expected — Phase 4 isn't designed to *outperform* Phase 3 on accuracy; it's designed to match accuracy while adding interpretability. The contribution is the per-step rationale, not the absolute number. (If pressed: the auxiliary multi-task losses also act as regularization — Phase 4 is more stable across seeds.)

**Q: How is this different from just an attention heatmap?**
A: A heatmap tells you where the model looked; our concept-gated trace tells you *along which reasoning dimension* it looked. Step 1 shows you what affective framing the model detected; Step 2 modal/hedging language; Step 3 negation. That structured decomposition is what makes the rationale human-readable.

**Q: Where does CoAID help and where doesn't it?**
A: Joint training boosts aggregate macro-F1 by ~9 points, almost entirely from the CoAID slice (which jumps to 0.95). The LIAR slice holds within 0.5 of the LIAR-only baseline — the joint-training tax. We use a domain-balanced sampler to keep this tax small.

**Q: What's the smallest input where it still works?**
A: We use L=128 word-tokens (DistilBERT subword max len 128). Most LIAR statements and CoAID claim titles fit comfortably.

**Q: Do you handle metadata for CoAID?**
A: CoAID rows are tagged `has_meta=0` and the MetaEncoder zeroes out its output for those rows. The fitted vocabularies and credit-history scaler are computed only on `has_meta=1` rows so CoAID's missing values can't poison them.

**Q: Why not use an LLM?**
A: An LLM is a black box for the prediction step — you can't point at attention weights to ground the rationale. Our model produces rationales that are causally tied to model internals, runs on a laptop, and costs no inference budget per query.

## 5 · Numbers worth memorizing

- LIAR: 12,836 political claims, 6-class → binarized fake/real (44/56).
- CoAID: ~3.5K claim-style entries (Claim + News headline subsets, after dedup).
- Joint corpus after merge + dedup: 13,254 rows · stratified 70/15/15 on (binary_label, dataset_source).
- DistilBERT: 66M params · 6 transformer layers · 768 hidden dim.
- Phase 3 head: ~2M trainable params on top of frozen BERT.
- Phase 4 head: ~3.4M trainable params (Phase 3 + 3 concept heads + fusion).

## 6 · If something breaks during the demo

- *Model file not found* → `ls models/` and check `phase4_merged_L128_best.pt` exists. If only `phase4_L128_best.pt` exists, run with `--tag phase4_L128 --dataset liar`.
- *spacy model missing* → `python -m spacy download en_core_web_sm`.
- *rationale file looks wrong* → re-run the curator with stricter filters: `python src/phase4/curate_rationales.py --min_concept_score 0.20 --n_examples 6`.

---

## Generate the curated examples (do this before the viva)

```bash
source venv/bin/activate
python src/phase4/curate_rationales.py --tag phase4_merged_L128
# writes logs/rationales_phase4/curated_for_viva.txt
```

Open the file. Read it once before the viva so you know which examples to point at first.
