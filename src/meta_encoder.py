"""
src/meta_encoder.py

Shared LIAR metadata encoder. Replaces the previous ad-hoc "party one-hot +
credit counts" treatment with a proper mixed-numeric/categorical encoder:

    Party         : 7-dim one-hot  (closed vocabulary)
    Credit history: 5 z-normalized counts
    Speaker       : top-K learned embedding (dim 32)
    Subject       : top-K learned embedding on FIRST subject (dim 16)
    Context       : top-K learned embedding (dim 16)

Why this matters:
  Karimi & Tang (2019) showed that the speaker's prior truthfulness record
  is the single most predictive LIAR feature — more than the statement itself.
  The previous pipeline only used the 5 aggregate credit counts; it ignored
  the identity of the speaker. Adding a top-500 speaker embedding routinely
  lifts LIAR binary accuracy by 1-2 points on top of a BERT feature model.

API:
    # 1. Fit vocabs + scaler on TRAIN ONLY (no leakage)
    vocabs, scaler = fit_metadata(train_df)

    # 2. Encode each split the same way
    train_meta = encode_metadata(train_df, vocabs, scaler)   # (N, 15) float32
    val_meta   = encode_metadata(val_df,   vocabs, scaler)
    test_meta  = encode_metadata(test_df,  vocabs, scaler)

    # 3. At model construction
    meta_enc = MetaEncoder(vocabs, dropout=0.4)              # out_dim=96

    # 4. In forward
    h_meta = meta_enc(meta_tensor)                           # (B, 96)

Tensor layout of (N, 15):
    cols  0-6  : party one-hot
    cols  7-11 : credit counts (z-normed)
    col   12   : speaker idx (float-cast long; 0 = UNK)
    col   13   : subject idx (first subject; 0 = UNK)
    col   14   : context idx (0 = UNK)
"""
from __future__ import annotations

from collections import Counter
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


PARTY_VOCAB = [
    "republican", "democrat", "independent",
    "libertarian", "green", "none", "unknown",
]
CREDIT_COLS = [
    "barely_true_count", "false_count", "half_true_count",
    "mostly_true_count", "pants_fire_count",
]

META_NUMERIC_DIM = len(PARTY_VOCAB) + len(CREDIT_COLS)   # 12
META_CAT_FIELDS  = ("speaker", "subject", "context")
META_TOTAL_DIM   = META_NUMERIC_DIM + len(META_CAT_FIELDS)  # 15


# ---------------------------------------------------------------------------
# Vocab fitting and encoding
# ---------------------------------------------------------------------------
def _norm(x) -> str:
    if pd.isna(x):
        return "unknown"
    return str(x).lower().strip()


def _first_subject(x) -> str:
    s = _norm(x)
    return s.split(",")[0].strip() if s else "unknown"


def _top_k_vocab(values, top_k: int) -> dict[str, int]:
    """{val -> idx in 1..K}. Index 0 is reserved for UNK."""
    c = Counter(values)
    common = [v for v, _ in c.most_common(top_k)]
    return {v: i + 1 for i, v in enumerate(common)}


def fit_metadata(
    train_df: pd.DataFrame,
    speaker_top_k: int = 500,
    subject_top_k: int = 100,
    context_top_k: int = 80,
) -> tuple[dict, dict]:
    """
    Fit the categorical vocabularies AND the credit-history scaler from the
    TRAIN split only. Returns (vocabs, scaler).

    vocabs = {
        "speaker": {name: idx},
        "subject": {name: idx},
        "context": {name: idx},
        "sizes":   {"speaker": int, "subject": int, "context": int},
          # sizes INCLUDE the UNK slot (idx 0), so embedding size = sizes[k]
    }

    scaler = {"mu": np.ndarray(5,), "sd": np.ndarray(5,)}
    """
    spk_vocab = _top_k_vocab([_norm(s) for s in train_df["speaker"]], speaker_top_k)
    sub_vocab = _top_k_vocab([_first_subject(s) for s in train_df["subject"]], subject_top_k)
    ctx_vocab = _top_k_vocab([_norm(s) for s in train_df["context"]], context_top_k)

    counts = train_df[CREDIT_COLS].fillna(0).astype(np.float32).values  # (N, 5)
    mu = counts.mean(axis=0)
    sd = counts.std(axis=0).clip(min=1e-6)

    vocabs = {
        "speaker": spk_vocab,
        "subject": sub_vocab,
        "context": ctx_vocab,
        "sizes": {
            "speaker": len(spk_vocab) + 1,
            "subject": len(sub_vocab) + 1,
            "context": len(ctx_vocab) + 1,
        },
    }
    scaler = {"mu": mu, "sd": sd}
    return vocabs, scaler


def encode_metadata(df: pd.DataFrame, vocabs: dict, scaler: dict) -> np.ndarray:
    """
    Build the (N, 15) float32 metadata tensor using pre-fit vocabs + scaler.
    Speakers/subjects/contexts not seen during fit → UNK idx 0.
    """
    N = len(df)

    # Party one-hot (closed 7-way vocab)
    parties = [_norm(p) for p in df["party"]]
    p_idx = [
        PARTY_VOCAB.index(p) if p in PARTY_VOCAB else PARTY_VOCAB.index("unknown")
        for p in parties
    ]
    p_onehot = np.eye(len(PARTY_VOCAB), dtype=np.float32)[p_idx]        # (N, 7)

    # Credit-history counts, z-normalized with train stats
    counts = df[CREDIT_COLS].fillna(0).astype(np.float32).values        # (N, 5)
    counts = (counts - scaler["mu"]) / scaler["sd"]

    # Categorical indices (float-cast so they fit one tensor)
    spk_idx = np.array(
        [vocabs["speaker"].get(_norm(s), 0) for s in df["speaker"]],
        dtype=np.float32,
    ).reshape(-1, 1)
    sub_idx = np.array(
        [vocabs["subject"].get(_first_subject(s), 0) for s in df["subject"]],
        dtype=np.float32,
    ).reshape(-1, 1)
    ctx_idx = np.array(
        [vocabs["context"].get(_norm(s), 0) for s in df["context"]],
        dtype=np.float32,
    ).reshape(-1, 1)

    meta = np.concatenate([p_onehot, counts, spk_idx, sub_idx, ctx_idx], axis=1)
    assert meta.shape == (N, META_TOTAL_DIM), meta.shape
    return meta.astype(np.float32)


def describe_vocabs(vocabs: dict) -> str:
    """Human-readable summary (for training log)."""
    s = vocabs["sizes"]
    return (
        f"speaker={s['speaker']-1}+UNK  "
        f"subject={s['subject']-1}+UNK  "
        f"context={s['context']-1}+UNK"
    )


# ---------------------------------------------------------------------------
# Neural encoder
# ---------------------------------------------------------------------------
class MetaEncoder(nn.Module):
    """
    Mixed numeric + learned-categorical encoder.

    Input  : (B, 15) float tensor  — layout as documented above.
    Output : (B, out_dim) float tensor.
    """

    def __init__(
        self,
        vocabs: dict,
        party_dim: int = len(PARTY_VOCAB),
        credit_dim: int = len(CREDIT_COLS),
        speaker_dim: int = 32,
        subject_dim: int = 16,
        context_dim: int = 16,
        out_dim: int = 96,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.party_dim  = party_dim
        self.credit_dim = credit_dim
        sizes = vocabs["sizes"]

        # padding_idx=0 keeps the UNK slot at exactly zero so unseen entities
        # can't leak residual signal through uninitialised weight noise.
        self.speaker_emb = nn.Embedding(sizes["speaker"], speaker_dim, padding_idx=0)
        self.subject_emb = nn.Embedding(sizes["subject"], subject_dim, padding_idx=0)
        self.context_emb = nn.Embedding(sizes["context"], context_dim, padding_idx=0)

        total_in = party_dim + credit_dim + speaker_dim + subject_dim + context_dim
        self.proj = nn.Sequential(
            nn.Linear(total_in, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.out_dim = out_dim

    def forward(self, meta: torch.Tensor) -> torch.Tensor:
        # Slice the packed tensor
        party  = meta[:, : self.party_dim]
        credit = meta[:, self.party_dim : self.party_dim + self.credit_dim]
        base   = self.party_dim + self.credit_dim
        spk    = meta[:, base + 0].long().clamp(min=0)
        sub    = meta[:, base + 1].long().clamp(min=0)
        ctx    = meta[:, base + 2].long().clamp(min=0)

        x = torch.cat(
            [
                party,
                credit,
                self.speaker_emb(spk),
                self.subject_emb(sub),
                self.context_emb(ctx),
            ],
            dim=-1,
        )
        return self.proj(x)
