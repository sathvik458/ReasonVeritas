"""
src/phase3/bilstm_attention.py

BiLSTM-Attention head trained on top of frozen DistilBERT features.

Inputs (per batch):
    features        : (B, T, 768)  pre-computed contextual embeddings
    attention_mask  : (B, T)       1 for real tokens, 0 for padding
    metadata        : (B, M) or None — credit-history counts + categorical embeddings

Architecture:
    features ─┐
              ├─ 2-layer BiLSTM (256 hidden, dropout between layers)
              │       │
              │       ▼
              │   Multi-head self-attention (4 heads, 512 dim) over time
              │       │
              │       ▼
              │   Mean+Max pooling over time, concat → (B, 1024)
              ▼
   [optional] metadata embedding ─ concat ─→
              │
              ▼
       Dropout(0.4) → Linear(→256) → GELU → Dropout(0.4) → Linear(→num_classes)

Returns logits and the attention weights (for rationale generation).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLSTMAttention(nn.Module):
    def __init__(
        self,
        bert_dim: int = 768,
        hidden_size: int = 256,
        num_lstm_layers: int = 2,
        num_attn_heads: int = 4,
        meta_dim: int = 0,
        meta_vocabs: dict | None = None,    # if set → use MetaEncoder
        num_classes: int = 2,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.bert_dim = bert_dim
        self.hidden_size = hidden_size
        self.bidir_dim = hidden_size * 2
        self.meta_dim = meta_dim
        self.meta_vocabs = meta_vocabs

        # Project BERT 768 → 2*hidden so the residual-style attention works cleanly.
        self.input_proj = nn.Linear(bert_dim, self.bidir_dim)
        self.input_dropout = nn.Dropout(dropout)

        # 2-layer BiLSTM with intra-stack dropout
        self.bilstm = nn.LSTM(
            input_size=self.bidir_dim,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
        )

        # Multi-head self-attention over time
        self.attn = nn.MultiheadAttention(
            embed_dim=self.bidir_dim,
            num_heads=num_attn_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(self.bidir_dim)

        # Pooled vector: mean + max → 2 * bidir_dim
        pooled_dim = self.bidir_dim * 2

        # Optional metadata branch
        if meta_vocabs is not None:
            # Lazy import so this file stays self-contained at definition time
            import os, sys
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from meta_encoder import MetaEncoder
            self.meta_proj = MetaEncoder(meta_vocabs, dropout=dropout)
            classifier_in = pooled_dim + self.meta_proj.out_dim
        elif meta_dim > 0:
            self.meta_proj = nn.Sequential(
                nn.Linear(meta_dim, 64),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            classifier_in = pooled_dim + 64
        else:
            self.meta_proj = None
            classifier_in = pooled_dim

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(classifier_in, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def encode(self, features: torch.Tensor, attention_mask: torch.Tensor):
        """
        features       : (B, T, 768)
        attention_mask : (B, T)  int/bool

        Returns:
            H            : (B, T, 2*hidden) post-LSTM hidden states
            attn_weights : (B, T, T) self-attention weights (averaged over heads)
            pooled       : (B, 2*pooled_dim) mean+max
        """
        x = self.input_proj(features)
        x = self.input_dropout(x)

        H, _ = self.bilstm(x)                                 # (B, T, 2H)

        # Multi-head attention with key-padding mask
        # `key_padding_mask` expects True where tokens should be IGNORED
        kpm = (attention_mask == 0)
        attn_out, attn_weights = self.attn(
            H, H, H,
            key_padding_mask=kpm,
            need_weights=True,
            average_attn_weights=True,
        )
        H = self.attn_norm(H + attn_out)                      # residual + LN

        # Mean-pool and max-pool over time (mask-aware)
        mask_f = attention_mask.unsqueeze(-1).float()         # (B, T, 1)
        masked = H * mask_f
        denom = mask_f.sum(dim=1).clamp(min=1.0)
        mean_pool = masked.sum(dim=1) / denom                 # (B, 2H)

        very_neg = torch.finfo(H.dtype).min
        max_input = H.masked_fill(mask_f == 0, very_neg)
        max_pool, _ = max_input.max(dim=1)                    # (B, 2H)

        pooled = torch.cat([mean_pool, max_pool], dim=-1)     # (B, 4H)
        return H, attn_weights, pooled

    def forward(
        self,
        features: torch.Tensor,
        attention_mask: torch.Tensor,
        metadata: torch.Tensor | None = None,
        has_meta: torch.Tensor | None = None,
    ):
        H, attn_weights, pooled = self.encode(features, attention_mask)

        if self.meta_proj is not None and metadata is not None:
            # MetaEncoder accepts a has_meta mask; the legacy Linear branch
            # does not. Detect by presence of meta_vocabs.
            if self.meta_vocabs is not None:
                m = self.meta_proj(metadata, has_meta=has_meta)
            else:
                m = self.meta_proj(metadata)
            pooled = torch.cat([pooled, m], dim=-1)

        logits = self.classifier(pooled)
        return logits, attn_weights, H
