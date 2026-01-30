import pandas as pd
import os
import ast

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
    "negation_count": 0
}

# -----------------------------
# CONCEPT MAPPING FUNCTION
# -----------------------------

def map_concepts(tokens):

    concepts = []

    for token in tokens:

        if token in EMOTION_WORDS:
            concepts.append("EMOTION")
            log["emotion_count"] += 1

        elif token in MODAL_WORDS:
            concepts.append("MODALITY")
            log["modality_count"] += 1

        elif token in NEGATION_WORDS:
            concepts.append("NEGATION")
            log["negation_count"] += 1

        else:
            concepts.append("O")  # Outside concept

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
