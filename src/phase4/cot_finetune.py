"""
phase4/cot_finetune.py

End-to-end fine-tuned DistilBERT + Chain-of-Thought reasoning model.

Wraps DistilBERT (top-N transformer layers trainable) in front of the
existing CoT architecture (BiLSTMAttention backbone + 3 concept-gated
attention heads + auxiliary multi-task losses). Same forward signature
as CoTModel except it consumes input_ids instead of pre-cached features.

Forward inputs:
    input_ids        : (B, T) long
    attention_mask   : (B, T) long/int
    emotion_mask     : (B, T) float — subword-aligned concept mask
    modality_mask    : (B, T) float
    negation_mask    : (B, T) float
    metadata         : (B, M) float or None

Returns the same 11-tuple as CoTModel:
    (logits,
     em_score, mo_score, ne_score,
     em_aux,   mo_aux,   ne_aux,
     em_alpha, mo_alpha, ne_alpha,
     base_attn)
"""
from __future__ import annotations

import os
import sys
import torch
import torch.nn as nn

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_root, "src"))
sys.path.insert(0, os.path.join(_root, "src", "phase3"))
sys.path.insert(0, os.path.join(_root, "src", "phase4"))
from config import BERT_MODEL_NAME
from cot_model import CoTModel, ConceptHead   # reuse existing concept-head logic
from bilstm_attention import BiLSTMAttention
from bert_finetune import configure_distilbert_freeze


class CoTModelFineTune(nn.Module):
    """
    DistilBERT (top-N unfrozen) → BiLSTMAttention backbone → 3 concept heads.

    Internally we instantiate a CoTModel with bert_dim=768 and call its
    `forward(features, ...)` after producing features from the live BERT.
    This keeps the head architecture (concept heads, fusion, classifier)
    100% identical to the frozen-feature pipeline so ablations are clean.
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
        dropout: float = 0.2,
    ):
        super().__init__()
        from transformers import AutoModel
        self.bert = AutoModel.from_pretrained(bert_model_name)
        configure_distilbert_freeze(self.bert, unfreeze_top_n)
        bert_dim = self.bert.config.hidden_size

        # Reuse the existing CoTModel exactly — it owns the BiLSTM backbone,
        # the three concept heads, and the fusion classifier.
        self.cot = CoTModel(
            bert_dim=bert_dim,
            hidden_size=hidden_size,
            num_lstm_layers=num_lstm_layers,
            num_attn_heads=num_attn_heads,
            meta_dim=meta_dim,
            meta_vocabs=meta_vocabs,
            num_classes=num_classes,
            dropout=dropout,
        )

    # Two-group LR helpers
    def bert_parameters(self):
        return [p for p in self.bert.parameters() if p.requires_grad]

    def head_parameters(self):
        return list(self.cot.parameters())

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        emotion_mask: torch.Tensor,
        modality_mask: torch.Tensor,
        negation_mask: torch.Tensor,
        metadata: torch.Tensor | None = None,
    ):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        features = out.last_hidden_state                  # (B, T, 768)
        return self.cot(
            features, attention_mask,
            emotion_mask, modality_mask, negation_mask,
            metadata,
        )
