import pandas as pd
import os
import ast
import sys

_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _src)
from utils_logger import update_log
from config import DATA_DIR, SEQUENCE_LENGTHS


# -----------------------------
# DEFINE CONCEPT LEXICONS
# -----------------------------

EMOTION_WORDS = {
    "shocking", "unbelievable", "outrageous", "disaster",
    "horrible", "terrible", "amazing", "incredible"
}

MODAL_WORDS = {
    "may", "might", "could", "would", "should", "must"
}

NEGATION_WORDS = {
    "not", "never", "no", "none", "cannot"
}


def map_concepts(tokens, log):
    tokens = list(tokens)
    concepts = []
    for token in tokens:
        token_lower = token.lower()
        log["total_tokens"] += 1
        if token_lower in EMOTION_WORDS:
            concepts.append("EMOTION")
            log["emotion_count"] += 1
        elif token_lower in MODAL_WORDS:
            concepts.append("MODALITY")
            log["modality_count"] += 1
        elif token_lower in NEGATION_WORDS:
            concepts.append("NEGATION")
            log["negation_count"] += 1
        else:
            concepts.append("O")
    return concepts


# -----------------------------
# PROCESS EACH L (128, 256, 512)
# -----------------------------

os.makedirs(DATA_DIR, exist_ok=True)

for L in SEQUENCE_LENGTHS:
    input_path = os.path.join(DATA_DIR, f"liar_truncated_L{L}.csv")
    df = pd.read_csv(input_path)
    df["truncated_tokens"] = df["truncated_tokens"].apply(ast.literal_eval)

    log = {"emotion_count": 0, "modality_count": 0, "negation_count": 0, "total_tokens": 0}
    df["concept_tags"] = df["truncated_tokens"].apply(lambda t: map_concepts(t, log))

    output_path = os.path.join(DATA_DIR, f"liar_concepts_step6_L{L}.csv")
    df.to_csv(output_path, index=False)

    emotion_ratio = log["emotion_count"] / log["total_tokens"] if log["total_tokens"] > 0 else 0
    modality_ratio = log["modality_count"] / log["total_tokens"] if log["total_tokens"] > 0 else 0
    negation_ratio = log["negation_count"] / log["total_tokens"] if log["total_tokens"] > 0 else 0

    update_log(f"Step6_ConceptMapping_L{L}", {
        "dataset_name": "LIAR",
        "total_samples": len(df),
        "total_tokens": log["total_tokens"],
        "emotion_tokens": log["emotion_count"],
        "modality_tokens": log["modality_count"],
        "negation_tokens": log["negation_count"],
        "emotion_ratio": round(emotion_ratio, 4),
        "modality_ratio": round(modality_ratio, 4),
        "negation_ratio": round(negation_ratio, 4)
    })

print("\nCONCEPT MAPPING SUMMARY (L = 128, 256, 512)")
print("Outputs: liar_concepts_step6_L128.csv, liar_concepts_step6_L256.csv, liar_concepts_step6_L512.csv")
