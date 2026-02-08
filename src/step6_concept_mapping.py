import pandas as pd
import os
import ast
from utils_logger import update_log


# -----------------------------
# LOAD TRUNCATED TOKEN DATA (STEP-4)
# -----------------------------

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
input_path = os.path.join(BASE_DIR, "data", "liar_truncated_L128.csv")

df = pd.read_csv(input_path)

# Convert string list back to Python list
df["truncated_tokens"] = df["truncated_tokens"].apply(ast.literal_eval)


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


# -----------------------------
# LOGGING
# -----------------------------

log = {
    "emotion_count": 0,
    "modality_count": 0,
    "negation_count": 0,
    "total_tokens": 0
}


# -----------------------------
# CONCEPT MAPPING FUNCTION
# -----------------------------

def map_concepts(tokens):

    tokens = list(tokens)  # safety
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
# APPLY CONCEPT MAPPING
# -----------------------------

df["concept_tags"] = df["truncated_tokens"].apply(map_concepts)


# -----------------------------
# SAVE OUTPUT
# -----------------------------

output_path = os.path.join(BASE_DIR, "data", "liar_concepts_step6.csv")
df.to_csv(output_path, index=False)


# -----------------------------
# PRINT SUMMARY
# -----------------------------

print("\nCONCEPT MAPPING SUMMARY")
print(f"Emotion tokens detected  : {log['emotion_count']}")
print(f"Modality tokens detected : {log['modality_count']}")
print(f"Negation tokens detected : {log['negation_count']}")


# -----------------------------
# COMPUTE RATIOS
# -----------------------------

emotion_ratio = log["emotion_count"] / log["total_tokens"] if log["total_tokens"] > 0 else 0
modality_ratio = log["modality_count"] / log["total_tokens"] if log["total_tokens"] > 0 else 0
negation_ratio = log["negation_count"] / log["total_tokens"] if log["total_tokens"] > 0 else 0


# -----------------------------
# CENTRAL LOG UPDATE
# -----------------------------

update_log("Step6_ConceptMapping", {
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
