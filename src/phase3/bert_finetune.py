"""
src/phase3/bert_finetune.py

End-to-end fine-tuned DistilBERT + BiLSTM-Attention classifier.

Architecture (vs. the frozen-feature variant):

    input_ids ─→ DistilBERT (bottom 4 transformer layers FROZEN,
                              top 2 layers TRAINABLE)
              ─→ last_hidden_state (B, T, 768)
              ─→ BiLSTMAttention head (same as Phase 3 frozen pipeline)
              ─→ logits

Why selective unfreezing:
  Fully fine-tuning all 6 DistilBERT layers on 9k LIAR examples will
  overfit fast and forget pre-trained knowledge ("catastrophic forgetting").
  Empirically (Howard & Ruder 2018; Sun et al. 2019) the top 2-3 transformer
  blocks carry most of the task-specific signal, so unfreezing only those
  gives most of the accuracy lift while keeping training stable.

Two-group LR:
  * BERT params:  2e-5 (standard fine-tuning rate, ULMFiT-style)
  * Head params:  5e-4 (the head trains from scratch — needs a bigger LR)
  Built into train_finetune.py.

API mirrors BiLSTMAttention so downstream code can swap models with minimal
changes; the signature differs only in input (input_ids instead of features).
"""
from __future__ import annotations

import os
import sys
import torch
import torch.nn as nn

_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _src)
sys.path.insert(0, os.path.join(_src, "phase3"))
from config import BERT_MODEL_NAME
from bilstm_attention import BiLSTMAttention


# ---------------------------------------------------------------------------
# Helper: freeze all bottom layers of DistilBERT, unfreeze the top N
# ---------------------------------------------------------------------------
def configure_distilbert_freeze(bert, unfreeze_top_n: int):
    """
    DistilBERT structure:
        bert.embeddings              (word + position embeddings)
        bert.transformer.layer[0..5] (6 transformer blocks)

    Always frozen:  embeddings (saves ~23M params, no measurable accuracy cost).
    Unfrozen:       the top N transformer layers.
    """
    # Freeze everything first
    for p in bert.parameters():
        p.requires_grad = False

    # Always keep embeddings frozen (they don't help much on small tasks)
    # and unfreeze the top-N transformer layers.
    n_layers = len(bert.transformer.layer)
    if unfreeze_top_n > n_layers:
        unfreeze_top_n = n_layers
    for i in range(n_layers - unfreeze_top_n, n_layers):
        for p in bert.transformer.layer[i].parameters():
            p.requires_grad = True

    # Helpful summary
    n_trainable = sum(p.numel() for p in bert.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in bert.parameters())
    print(
        f"[bert_finetune] DistilBERT layers={n_layers}, "
        f"unfrozen top {unfreeze_top_n} → "
        f"{n_trainable:,} / {n_total:,} BERT params trainable "
        f"({100*n_trainable/n_total:.1f}%)"
    )


# ---------------------------------------------------------------------------
# Fine-tuned model
# ---------------------------------------------------------------------------
class FineTunedBERTClassifier(nn.Module):
    """
    DistilBERT (top-N layers trainable) + BiLSTMAttention head.

    Forward inputs:
        input_ids       : (B, T) long
        attention_mask  : (B, T) long/int (1=real, 0=pad)
        metadata        : (B, M) float or None

    Returns:
        logits          : (B, num_classes)
        attn_weights    : (B, T, T) head's self-attention (averaged over heads)
        H               : (B, T, 2*hidden_size) head's BiLSTM hidden states
    """

    def __init__(
        self,
        bert_model_name: str = BERT_MODEL_NAME,
        unfreeze_top_n: int = 2,
        hidden_size: int = 192,
        num_lstm_layers: int = 1,
        num_attn_heads: int = 4,
        meta_dim: int = 0,
        meta_vocabs: dict | None = None,
        num_classes: int = 2,
        dropout: float = 0.2,            # lower than frozen — BERT brings its own regularization
    ):
        super().__init__()
        from transformers import AutoModel
        self.bert = AutoModel.from_pretrained(bert_model_name)
        configure_distilbert_freeze(self.bert, unfreeze_top_n)
        bert_dim = self.bert.config.hidden_size  # 768

        self.head = BiLSTMAttention(
            bert_dim=bert_dim,
            hidden_size=hidden_size,
            num_lstm_layers=num_lstm_layers,
            num_attn_heads=num_attn_heads,
            meta_dim=meta_dim,
            meta_vocabs=meta_vocabs,
            num_classes=num_classes,
            dropout=dropout,
        )

    # Convenience accessors so trainer can build two LR groups
    def bert_parameters(self):
        return [p for p in self.bert.parameters() if p.requires_grad]

    def head_parameters(self):
        return list(self.head.parameters())

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        metadata: torch.Tensor | None = None,
        has_meta: torch.Tensor | None = None,
    ):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        features = out.last_hidden_state                      # (B, T, 768)
        return self.head(features, attention_mask, metadata, has_meta=has_meta)
