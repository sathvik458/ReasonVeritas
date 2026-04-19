"""
src/bert_features.py

Frozen DistilBERT feature extractor with on-disk caching.

Used by Phase 2 step 3 (one-time pre-computation) and by Phase 3 / Phase 4
training scripts (cached lookup, no recomputation).

Design choices:
  * DistilBERT-base-uncased (66 M params) — chosen for M2/8 GB unified memory.
    768-dim hidden states, ~4× faster than BERT-base on Apple Silicon.
  * Frozen — no fine-tuning. We cache token-level hidden states once and train
    a light head on top. This is the standard CPU/MPS-friendly setup and
    keeps each ablation run cheap.
  * Token-level features are kept (not pooled) so Phase 4 concept-gated
    attention can address individual subword positions.
  * A `word_offsets` array is also returned — for each input word index, it
    gives (subword_start, subword_end) within the BERT sequence.  Phase 4
    uses this to map word-level concept tags to subword-level masks.
"""
from __future__ import annotations

import os
import sys
import numpy as np
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset

_src = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _src)
from config import BERT_MODEL_NAME, BERT_BATCH_SIZE, get_device

_TOKENIZER = None
_MODEL = None
_DEVICE = None


def _load():
    """Lazily load tokenizer + model. Cached at module level."""
    global _TOKENIZER, _MODEL, _DEVICE
    if _MODEL is not None:
        return _TOKENIZER, _MODEL, _DEVICE
    from transformers import AutoTokenizer, AutoModel
    _DEVICE = get_device()
    print(f"[bert_features] loading {BERT_MODEL_NAME} on {_DEVICE} ...")
    _TOKENIZER = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    _MODEL = AutoModel.from_pretrained(BERT_MODEL_NAME)
    _MODEL.eval()
    _MODEL.to(_DEVICE)
    for p in _MODEL.parameters():
        p.requires_grad = False
    return _TOKENIZER, _MODEL, _DEVICE


def encode_words(
    word_lists: List[List[str]],
    bert_max_len: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run frozen DistilBERT over a list of pre-tokenized sentences (lists of words).

    Returns (all on CPU, numpy):
        features        : float32 (N, bert_max_len, 768)  — last hidden states
        attention_mask  : int8    (N, bert_max_len)       — 1 = real, 0 = pad
        subword_to_word : int32   (N, bert_max_len)       — for each subword,
                                                            the index of the
                                                            originating word (-1 if special / pad)
        n_words         : int32   (N,)                    — number of words kept
                                                            (after subword truncation)
    """
    tokenizer, model, device = _load()

    n = len(word_lists)
    features = np.zeros((n, bert_max_len, model.config.hidden_size), dtype=np.float32)
    attn     = np.zeros((n, bert_max_len), dtype=np.int8)
    sw2w     = np.full((n, bert_max_len), -1, dtype=np.int32)
    n_words  = np.zeros((n,), dtype=np.int32)

    # Process in chunks to bound memory
    for start in range(0, n, BERT_BATCH_SIZE):
        end = min(start + BERT_BATCH_SIZE, n)
        chunk = word_lists[start:end]

        # is_split_into_words=True tells the tokenizer the input is already
        # word-tokenised; it returns word_ids() per subword for alignment.
        enc = tokenizer(
            chunk,
            is_split_into_words=True,
            padding="max_length",
            truncation=True,
            max_length=bert_max_len,
            return_tensors="pt",
        )

        input_ids = enc["input_ids"].to(device)
        amask     = enc["attention_mask"].to(device)

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=amask)
            hidden = out.last_hidden_state  # (B, T, 768)

        features[start:end] = hidden.cpu().numpy()
        attn[start:end]     = amask.cpu().numpy().astype(np.int8)

        # Build subword-to-word index
        for i in range(end - start):
            wids = enc.word_ids(batch_index=i)  # list of len bert_max_len
            seen_words = set()
            for j, wid in enumerate(wids):
                if wid is None:
                    sw2w[start + i, j] = -1
                else:
                    sw2w[start + i, j] = wid
                    seen_words.add(wid)
            n_words[start + i] = len(seen_words)

        if (start // BERT_BATCH_SIZE) % 10 == 0:
            print(f"  [bert_features] {end}/{n}")

    return features, attn, sw2w, n_words


def cache_features(
    word_lists: List[List[str]],
    cache_path: str,
    bert_max_len: int,
    extras: dict | None = None,
    overwrite: bool = False,
):
    """
    Compute and save features to .npz. If file exists and overwrite=False,
    skip. `extras` is an optional dict of additional arrays (e.g. labels)
    to bundle into the same .npz.
    """
    if os.path.exists(cache_path) and not overwrite:
        print(f"[bert_features] cache exists, skipping: {cache_path}")
        return
    features, attn, sw2w, n_words = encode_words(word_lists, bert_max_len)

    payload = dict(
        features=features,
        attention_mask=attn,
        subword_to_word=sw2w,
        n_words=n_words,
    )
    if extras:
        for k, v in extras.items():
            payload[k] = np.asarray(v)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(cache_path, **payload)
    print(f"[bert_features] wrote {cache_path}  ({features.shape} float32)")


def load_features(cache_path: str) -> dict:
    """Load a previously-cached .npz; returns a dict of arrays."""
    data = np.load(cache_path, allow_pickle=False)
    return {k: data[k] for k in data.files}


def word_mask_to_subword_mask(word_mask: np.ndarray, sw2w: np.ndarray) -> np.ndarray:
    """
    Project a word-level binary mask (length = num_words for one example) onto
    a subword-level mask (length = bert_max_len) using the subword-to-word
    index array.  Used by Phase 4 concept_masks.

    word_mask : (W,) 0/1
    sw2w      : (T,) int — word index per subword (-1 for special/pad)
    returns   : (T,) 0/1
    """
    sub = np.zeros_like(sw2w, dtype=np.float32)
    valid = sw2w >= 0
    valid_indices = sw2w[valid]
    # Clip in case of subword truncation past num_words
    safe = valid_indices < len(word_mask)
    valid_indices = valid_indices[safe]
    sub_idx = np.where(valid)[0][safe]
    sub[sub_idx] = word_mask[valid_indices]
    return sub
