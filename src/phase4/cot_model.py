"""
phase4/cot_model.py  — REWORKED for frozen-BERT backbone

Chain-of-Thought model on top of Phase 3's BiLSTMAttention (which now
consumes pre-computed DistilBERT features).

What this adds on top of Phase 3:
  - Three concept-gated attention heads: Emotion, Modality, Negation
  - Each head attends ONLY to tokens whose subword-level mask is 1
  - Each head outputs:
      * concept_score   (0..1)          — "how present is this concept?"
      * concept_context (B, bidir_dim)  — weighted sum of backbone hidden states
      * aux_logits      (B, num_classes)— independent classifier for multi-task
  - Final classifier fuses: pooled backbone + concept scores + concept contexts
    + optional metadata → logits

Why this is novel vs. post-hoc explanation methods:
  1. Concept reasoning is BUILT INTO the model via parallel gated attention
     (not derived post-hoc). Each head's mask is a linguistic commitment
     (emotion word / modal cue + scope / negation cue + scope).
  2. Auxiliary multi-task losses force each head to be individually
     predictive — reasoning is causally connected to the verdict.
  3. Attention-grounded rationales (in rationale.py) pull top-k tokens
     from each head's attention → faithful by construction.

Forward signature:
    forward(features, attention_mask, emotion_mask, modality_mask,
            negation_mask, metadata=None)

Returns:
    final_logits  (B, C)
    em_score, mo_score, ne_score  each (B, 1)
    em_aux, mo_aux, ne_aux        each (B, C)
    em_attn, mo_attn, ne_attn     each (B, T)  per-token attention weights
    base_attn                     (B, T, T)    backbone self-attention
"""
from __future__ import annotations

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_root, "src"))
sys.path.insert(0, os.path.join(_root, "src", "phase3"))
from bilstm_attention import BiLSTMAttention


class ConceptHead(nn.Module):
    """
    Gated attention over tokens belonging to ONE concept channel.

    Takes:
        H    : (B, T, D)  — backbone post-LSTM+attention hidden states
        mask : (B, T)     — binary float, 1 where this concept's cue/scope applies

    Returns:
        concept_score   : (B, 1) ∈ [0, 1]  — "how present is this concept?"
        concept_context : (B, D)           — weighted sum of H over mask
        aux_logits      : (B, C)           — independent prediction
        attn_weights    : (B, T)           — per-token attention (for rationales)
    """

    def __init__(self, hidden_dim: int, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        # 1-hop attention scorer
        self.attn_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attn_vec  = nn.Linear(hidden_dim, 1, bias=False)
        # Concept presence scorer
        self.presence  = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        # Auxiliary classifier — independent prediction from this head alone
        self.aux_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, H: torch.Tensor, mask: torch.Tensor):
        # mask : (B, T) float
        present = (mask.sum(dim=1, keepdim=True) > 0).float()        # (B, 1)

        # Additive attention scores
        u = torch.tanh(self.attn_proj(H))                            # (B, T, D)
        scores = self.attn_vec(u).squeeze(-1)                        # (B, T)

        # Gate attention to masked positions only. For examples with NO
        # concept tokens, we still need to produce a well-formed vector.
        very_neg = torch.finfo(scores.dtype).min
        gated = scores.masked_fill(mask == 0, very_neg)
        # If mask is all-zero, softmax over all -inf → NaN; guard with eye fallback.
        has_any = present.squeeze(-1) > 0                            # (B,) bool
        if has_any.all():
            alpha = F.softmax(gated, dim=1)
        else:
            alpha = torch.zeros_like(scores)
            if has_any.any():
                alpha[has_any] = F.softmax(gated[has_any], dim=1)

        # Context = weighted sum of H using gated alpha
        context = torch.sum(H * alpha.unsqueeze(-1), dim=1)          # (B, D)
        # Zero-context where concept is absent (downstream sees zeros → score→0)
        context = context * present

        # Concept-presence score in [0, 1]
        concept_score = torch.sigmoid(self.presence(context))
        concept_score = concept_score * present                      # enforce 0 if absent

        # Auxiliary classifier (trained only where concept is present to avoid
        # gradient noise from all-zero contexts)
        aux_logits = self.aux_classifier(context)

        return concept_score, context, aux_logits, alpha


class CoTModel(nn.Module):
    """
    Full reasoning model. Wraps the new BiLSTMAttention (BERT-feature backbone).

    Concept channels (3):  Emotion, Modality (+ scope), Negation (+ scope).

    Forward:
        features        : (B, T, bert_dim) cached BERT features
        attention_mask  : (B, T) int/bool
        emotion_mask    : (B, T) float
        modality_mask   : (B, T) float
        negation_mask   : (B, T) float
        metadata        : (B, M) float or None
    """

    def __init__(
        self,
        bert_dim: int = 768,
        hidden_size: int = 192,           # 2*hidden_size is the bidir dim
        num_lstm_layers: int = 1,
        num_attn_heads: int = 4,
        meta_dim: int = 0,
        meta_vocabs: dict | None = None,  # if set → use MetaEncoder (speaker/subject/context)
        num_classes: int = 2,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.backbone = BiLSTMAttention(
            bert_dim=bert_dim,
            hidden_size=hidden_size,
            num_lstm_layers=num_lstm_layers,
            num_attn_heads=num_attn_heads,
            meta_dim=0,                # we handle metadata HERE (not in backbone)
            meta_vocabs=None,
            num_classes=num_classes,
            dropout=dropout,
        )
        bidir = hidden_size * 2        # 384

        self.emotion_head  = ConceptHead(bidir, num_classes, dropout=dropout)
        self.modality_head = ConceptHead(bidir, num_classes, dropout=dropout)
        self.negation_head = ConceptHead(bidir, num_classes, dropout=dropout)

        # Fusion: pooled backbone (4H = 2*bidir) + 3 concept contexts (3*bidir)
        # + 3 concept scores + optional metadata
        self.meta_dim = meta_dim
        self.meta_vocabs = meta_vocabs
        if meta_vocabs is not None:
            import os, sys
            sys.path.insert(0, os.path.join(_root, "src"))
            from meta_encoder import MetaEncoder
            self.meta_proj = MetaEncoder(meta_vocabs, dropout=dropout)
            fusion_in = 2 * bidir + 3 * bidir + 3 + self.meta_proj.out_dim
        elif meta_dim > 0:
            self.meta_proj = nn.Sequential(
                nn.Linear(meta_dim, 64),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            fusion_in = 2 * bidir + 3 * bidir + 3 + 64
        else:
            self.meta_proj = None
            fusion_in = 2 * bidir + 3 * bidir + 3

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fusion_in, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(
        self,
        features: torch.Tensor,
        attention_mask: torch.Tensor,
        emotion_mask: torch.Tensor,
        modality_mask: torch.Tensor,
        negation_mask: torch.Tensor,
        metadata: torch.Tensor | None = None,
        has_meta: torch.Tensor | None = None,
    ):
        # Backbone encode → post-LSTM+attention H and mean+max pool
        H, base_attn, pooled = self.backbone.encode(features, attention_mask)

        # Concept heads (each gated by its own mask)
        em_score, em_ctx, em_aux, em_alpha = self.emotion_head(H, emotion_mask)
        mo_score, mo_ctx, mo_aux, mo_alpha = self.modality_head(H, modality_mask)
        ne_score, ne_ctx, ne_aux, ne_alpha = self.negation_head(H, negation_mask)

        # Fuse
        parts = [pooled, em_ctx, mo_ctx, ne_ctx, em_score, mo_score, ne_score]
        if self.meta_proj is not None and metadata is not None:
            # MetaEncoder accepts has_meta mask; legacy Linear branch does not.
            if self.meta_vocabs is not None:
                parts.append(self.meta_proj(metadata, has_meta=has_meta))
            else:
                parts.append(self.meta_proj(metadata))
        fused = torch.cat(parts, dim=-1)
        logits = self.classifier(fused)

        return (
            logits,
            em_score, mo_score, ne_score,
            em_aux, mo_aux, ne_aux,
            em_alpha, mo_alpha, ne_alpha,
            base_attn,
        )
