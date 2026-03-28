"""
phase4/cot_model.py

Chain-of-Thought model wrapping Phase 3's BiLSTMAttention.

What this adds on top of Phase 3:
  - Three concept attention heads: Emotion, Modality, Negation
  - Each head focuses ONLY on tokens tagged with its concept
  - Each head produces a concept score (0.0 to 1.0)
  - These three scores + the original context vector → final prediction
  - During training, each head gets its OWN auxiliary loss (Method C)

Why auxiliary losses make reasoning real:
  Without them, the model could ignore the concept heads entirely
  and just use the BiLSTM to memorize patterns. The auxiliary losses
  force each head to be genuinely predictive on its own, so the
  reasoning steps are causally connected to the verdict — not decorative.

Architecture:
  Input tokens
      ↓
  BiLSTMAttention (Phase 3) → context vector + alpha weights
      ↓
  Concept Head: Emotion   → emotion_score  → auxiliary loss
  Concept Head: Modality  → modality_score → auxiliary loss
  Concept Head: Negation  → negation_score → auxiliary loss
      ↓
  Fusion layer: [context | emotion_score | modality_score | negation_score]
      ↓
  Final classifier → logits + confidence
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# So we can import BiLSTMAttention from Phase 3
_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_root, "src", "phase3"))
from bilstm_attention import BiLSTMAttention


class ConceptHead(nn.Module):
    """
    A single concept attention head.

    It takes the BiLSTM hidden states (H) and a binary mask
    showing which tokens belong to this concept type.

    If NO tokens in the sequence are tagged (mask is all zeros),
    the head returns a score of 0.0 — this concept is absent.

    If concept tokens exist, it runs a small attention over ONLY
    those tokens and produces a single score between 0 and 1
    representing how strongly this concept is present.

    Args:
        hidden_size: the BiLSTM hidden size (128 in your config)
                     Note: BiLSTM is bidirectional so output is hidden_size * 2 = 256
    """
    def __init__(self, hidden_size):
        super(ConceptHead, self).__init__()
        bilstm_output_dim = hidden_size * 2  # 256 for your config

        # Small linear layer to compute attention scores over concept tokens
        self.attn = nn.Linear(bilstm_output_dim, 1, bias=False)

        # Projects the concept context vector to a single scalar score
        self.scorer = nn.Linear(bilstm_output_dim, 1)

        # Projects concept score to num_classes for auxiliary loss
        # This is what lets each head make its OWN prediction
        self.aux_classifier = nn.Linear(bilstm_output_dim, 2)

    def forward(self, H, mask):
        """
        Args:
            H    : BiLSTM hidden states, shape (batch, seq_len, hidden*2)
            mask : concept binary mask, shape (batch, seq_len)
                   1 where concept token exists, 0 elsewhere

        Returns:
            concept_score  : scalar per example, shape (batch, 1)
                             how strongly this concept is present
            concept_context: the weighted hidden state for this concept
                             shape (batch, hidden*2)
            aux_logits     : raw prediction logits from this head alone
                             shape (batch, 2) — used for auxiliary loss
        """
        # ------------------------------------------------------------------
        # Step 1: compute raw attention scores over all tokens
        # shape: (batch, seq_len, 1) → squeeze → (batch, seq_len)
        # ------------------------------------------------------------------
        raw_scores = self.attn(H).squeeze(-1)

        # ------------------------------------------------------------------
        # Step 2: mask out non-concept tokens by setting their score to -inf
        # After softmax, -inf → 0, so only concept tokens get attention weight
        # If mask is all zeros (no concept tokens), scores stay as-is
        # ------------------------------------------------------------------
        concept_present = mask.sum(dim=1, keepdim=True) > 0  # (batch, 1) bool

        masked_scores = raw_scores.masked_fill(mask == 0, -1e9)

        # Where no concept tokens exist, use uniform attention (fallback)
        uniform = torch.full_like(raw_scores, -1e9)
        final_scores = torch.where(concept_present, masked_scores, uniform)

        # ------------------------------------------------------------------
        # Step 3: softmax → attention weights over concept tokens only
        # shape: (batch, seq_len)
        # ------------------------------------------------------------------
        alpha = F.softmax(final_scores, dim=1)

        # Handle all-zero mask: set alpha to zero vector in that case
        alpha = alpha * concept_present.float()

        # ------------------------------------------------------------------
        # Step 4: weighted sum of hidden states → concept context vector
        # shape: (batch, hidden*2)
        # ------------------------------------------------------------------
        concept_context = torch.sum(H * alpha.unsqueeze(-1), dim=1)

        # ------------------------------------------------------------------
        # Step 5: score how strongly this concept is present (0.0 to 1.0)
        # shape: (batch, 1)
        # ------------------------------------------------------------------
        concept_score = torch.sigmoid(self.scorer(concept_context))

        # Zero out score where concept was absent
        concept_score = concept_score * concept_present.float()

        # ------------------------------------------------------------------
        # Step 6: auxiliary classifier — this head's own prediction
        # shape: (batch, 2)
        # This is used ONLY during training for the auxiliary loss
        # ------------------------------------------------------------------
        aux_logits = self.aux_classifier(concept_context)

        return concept_score, concept_context, aux_logits


class CoTModel(nn.Module):
    """
    Full Chain-of-Thought model.

    Wraps Phase 3's BiLSTMAttention and adds three concept heads
    with a fusion layer for the final prediction.

    Args:
        vocab_size  : vocabulary size (from Phase 2)
        embed_dim   : embedding dimension (300 in your config)
        hidden_size : BiLSTM hidden size (128 in your config)
        num_classes : 2 (fake / real)
    """
    def __init__(self, vocab_size, embed_dim=300, hidden_size=128, num_classes=2):
        super(CoTModel, self).__init__()

        # ---- Phase 3 backbone (reused exactly as-is) ----
        self.backbone = BiLSTMAttention(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_size=hidden_size,
            num_classes=num_classes
        )

        bilstm_output_dim = hidden_size * 2  # 256

        # ---- Three concept heads (Phase 4) ----
        self.emotion_head  = ConceptHead(hidden_size)
        self.modality_head = ConceptHead(hidden_size)
        self.negation_head = ConceptHead(hidden_size)

        # ---- Fusion classifier ----
        # Input: context vector (256) + 3 concept scores (1 each) = 259
        fusion_input_dim = bilstm_output_dim + 3  # 259
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, input_ids, emotion_mask, modality_mask, negation_mask):
        """
        Args:
            input_ids     : token indices, shape (batch, seq_len)
            emotion_mask  : binary float tensor, shape (batch, seq_len)
            modality_mask : binary float tensor, shape (batch, seq_len)
            negation_mask : binary float tensor, shape (batch, seq_len)

        Returns:
            final_logits   : shape (batch, 2) — main prediction
            emotion_score  : shape (batch, 1) — reasoning step 1
            modality_score : shape (batch, 1) — reasoning step 2
            negation_score : shape (batch, 1) — reasoning step 3
            aux_logits_e   : shape (batch, 2) — emotion head's own prediction
            aux_logits_m   : shape (batch, 2) — modality head's own prediction
            aux_logits_n   : shape (batch, 2) — negation head's own prediction
            base_alpha     : shape (batch, seq_len) — Phase 3 attention weights
        """
        # ------------------------------------------------------------------
        # Step 1: Run Phase 3 backbone
        # Gets us the base logits AND the BiLSTM hidden states
        # We need to reach inside backbone to get H (hidden states)
        # ------------------------------------------------------------------
        x = self.backbone.embedding(input_ids)           # (B, T, embed_dim)
        H, _ = self.backbone.bilstm(x)                   # (B, T, hidden*2)

        # Phase 3 attention (base alpha)
        score = torch.tanh(self.backbone.attn_linear(H))
        score = self.backbone.attn_vector(score).squeeze(-1)
        base_alpha = F.softmax(score, dim=1)
        context = torch.sum(H * base_alpha.unsqueeze(-1), dim=1)  # (B, hidden*2)

        # ------------------------------------------------------------------
        # Step 2: Run each concept head
        # Each head focuses only on its concept-tagged tokens
        # ------------------------------------------------------------------
        emotion_score,  _, aux_logits_e = self.emotion_head(H,  emotion_mask)
        modality_score, _, aux_logits_m = self.modality_head(H, modality_mask)
        negation_score, _, aux_logits_n = self.negation_head(H, negation_mask)

        # ------------------------------------------------------------------
        # Step 3: Fuse context + three scores into final prediction
        # This is where the reasoning causally affects the verdict
        # ------------------------------------------------------------------
        fused = torch.cat([context, emotion_score, modality_score, negation_score], dim=1)
        final_logits = self.fusion(fused)

        return (
            final_logits,
            emotion_score, modality_score, negation_score,
            aux_logits_e, aux_logits_m, aux_logits_n,
            base_alpha
        )