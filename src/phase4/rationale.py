print("rationale.py loaded")

"""
phase4/rationale.py

Generates human-readable Chain-of-Thought reasoning text
from the concept scores produced by the CoT model.

The scores are causally real (driven by auxiliary losses during training),
so this text reflects genuine model reasoning — not post-hoc decoration.

Example output:
  Step 1 [Emotion]:   score 0.74 → HIGH   — strong emotional language detected
  Step 2 [Modality]:  score 0.21 → low    — little hedging or uncertainty
  Step 3 [Negation]:  score 0.61 → HIGH   — contradictory negation patterns found
  ─────────────────────────────────────────
  Verdict:     FAKE
  Confidence:  79.3%
  ─────────────────────────────────────────
  Reasoning: This claim contains strong emotional language and negation
  patterns, which are common indicators of misleading content.
"""

import torch
import torch.nn.functional as F


# -----------------------------------------------------------------------
# Thresholds for labelling a concept score as HIGH / medium / low
# These are tunable — adjust based on your validation set score distributions
# -----------------------------------------------------------------------
HIGH_THRESHOLD   = 0.55   # score >= 0.55 → HIGH presence
MEDIUM_THRESHOLD = 0.30   # score >= 0.30 → medium presence
                           # score <  0.30 → low / absent


def score_label(score_value):
    """Returns a text label for a concept score float."""
    if score_value >= HIGH_THRESHOLD:
        return "HIGH"
    elif score_value >= MEDIUM_THRESHOLD:
        return "medium"
    else:
        return "low"


# -----------------------------------------------------------------------
# Interpretation templates per concept × level
# Describes what each score level means for fake news detection
# -----------------------------------------------------------------------
EMOTION_INTERPRETATIONS = {
    "HIGH"  : "strong emotional language detected — common in manipulative claims",
    "medium": "some emotional language present — mild signal",
    "low"   : "minimal emotional language — claim is relatively neutral",
}

MODALITY_INTERPRETATIONS = {
    "HIGH"  : "heavy use of hedging words (may/might/could) — claim avoids commitment",
    "medium": "some uncertainty markers present — partial hedging",
    "low"   : "claim is mostly direct and assertive — low hedging",
}

NEGATION_INTERPRETATIONS = {
    "HIGH"  : "strong negation patterns (not/never/none) — contradictory denials present",
    "medium": "some negation present — partial contradictions",
    "low"   : "little to no negation — claim is largely positive",
}


def generate_rationale(
    emotion_score,
    modality_score,
    negation_score,
    final_logits,
    label_map_inv=None
):
    """
    Generates a Chain-of-Thought reasoning string for ONE example.

    Args:
        emotion_score  : float, 0.0–1.0 (from model output)
        modality_score : float, 0.0–1.0
        negation_score : float, 0.0–1.0
        final_logits   : tensor of shape (2,) — raw model output for one example
        label_map_inv  : dict mapping int → label string, e.g. {0: "FAKE", 1: "REAL"}
                         defaults to {0: "FAKE", 1: "REAL"}

    Returns:
        rationale_text : str — the full CoT reasoning block
        verdict        : str — "FAKE" or "REAL"
        confidence     : float — model confidence in its verdict (0–100%)
    """
    if label_map_inv is None:
        label_map_inv = {0: "FAKE", 1: "REAL"}

    # ---- Compute verdict and confidence ----
    probs      = F.softmax(final_logits, dim=-1)           # [p_fake, p_real]
    pred_class = torch.argmax(probs).item()
    confidence = probs[pred_class].item() * 100.0
    verdict    = label_map_inv[pred_class]

    # ---- Score labels ----
    e_label = score_label(emotion_score)
    m_label = score_label(modality_score)
    n_label = score_label(negation_score)

    # ---- Interpretations ----
    e_text = EMOTION_INTERPRETATIONS[e_label]
    m_text = MODALITY_INTERPRETATIONS[m_label]
    n_text = NEGATION_INTERPRETATIONS[n_label]

    # ---- Build summary sentence ----
    # Identifies which concepts are the strongest drivers of the verdict
    drivers = []
    if e_label == "HIGH":
        drivers.append("emotional language")
    if n_label == "HIGH":
        drivers.append("negation patterns")
    if m_label == "HIGH":
        drivers.append("hedging language")

    if drivers:
        driver_str = " and ".join(drivers)
        if verdict == "FAKE":
            summary = (
                f"This claim contains {driver_str}, "
                f"which are common indicators of misleading content."
            )
        else:
            summary = (
                f"Despite some {driver_str}, "
                f"the overall pattern is consistent with credible content."
            )
    else:
        if verdict == "FAKE":
            summary = (
                "No dominant concept signals found, but the overall linguistic "
                "pattern matches fake news characteristics in the training data."
            )
        else:
            summary = (
                "The claim shows neutral linguistic patterns "
                "consistent with factual reporting."
            )

    # ---- Assemble full rationale block ----
    sep = "─" * 45
    rationale = (
        f"Step 1 [Emotion]:   score {emotion_score:.2f} → {e_label:<6}  — {e_text}\n"
        f"Step 2 [Modality]:  score {modality_score:.2f} → {m_label:<6}  — {m_text}\n"
        f"Step 3 [Negation]:  score {negation_score:.2f} → {n_label:<6}  — {n_text}\n"
        f"{sep}\n"
        f"Verdict:     {verdict}\n"
        f"Confidence:  {confidence:.1f}%\n"
        f"{sep}\n"
        f"Reasoning: {summary}"
    )

    return rationale, verdict, confidence


def generate_batch_rationales(
    emotion_scores,
    modality_scores,
    negation_scores,
    final_logits_batch,
    label_map_inv=None
):
    """
    Generates rationales for a whole batch.

    Args:
        emotion_scores       : tensor (batch,) or (batch, 1)
        modality_scores      : tensor (batch,) or (batch, 1)
        negation_scores      : tensor (batch,) or (batch, 1)
        final_logits_batch   : tensor (batch, 2)
        label_map_inv        : {0: "FAKE", 1: "REAL"}

    Returns:
        list of (rationale_text, verdict, confidence) tuples
    """
    # Flatten (batch, 1) → (batch,)
    e = emotion_scores.squeeze(-1).tolist()
    m = modality_scores.squeeze(-1).tolist()
    n = negation_scores.squeeze(-1).tolist()

    results = []
    for i in range(len(e)):
        r, v, c = generate_rationale(
            emotion_score=e[i],
            modality_score=m[i],
            negation_score=n[i],
            final_logits=final_logits_batch[i],
            label_map_inv=label_map_inv
        )
        results.append((r, v, c))
    return results