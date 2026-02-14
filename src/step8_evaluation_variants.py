import pandas as pd
import os
import ast
from utils_logger import update_log


# -----------------------------
# LOAD SPLIT DATA (TRAIN SET)
# -----------------------------

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
input_path = os.path.join(BASE_DIR, "data", "train_split.csv")

df = pd.read_csv(input_path)

# Convert string lists back to Python lists
df["truncated_tokens"] = df["truncated_tokens"].apply(ast.literal_eval)
df["concept_tags"] = df["concept_tags"].apply(ast.literal_eval)


# -----------------------------
# SUFFICIENCY VARIANT
# Keep only concept tokens
# -----------------------------

def keep_concepts(tokens, concepts):
    return [token for token, concept in zip(tokens, concepts) if concept != "O"]


# -----------------------------
# COMPREHENSIVENESS VARIANT
# Remove concept tokens
# -----------------------------

def remove_concepts(tokens, concepts):
    return [token for token, concept in zip(tokens, concepts) if concept == "O"]


# -----------------------------
# APPLY TRANSFORMATIONS
# -----------------------------

df["sufficiency_tokens"] = df.apply(
    lambda row: keep_concepts(row["truncated_tokens"], row["concept_tags"]),
    axis=1
)

df["comprehensiveness_tokens"] = df.apply(
    lambda row: remove_concepts(row["truncated_tokens"], row["concept_tags"]),
    axis=1
)


# -----------------------------
# COMPUTE STATISTICS
# -----------------------------

avg_suff_len = df["sufficiency_tokens"].apply(len).mean()
avg_comp_len = df["comprehensiveness_tokens"].apply(len).mean()
samples_no_concepts = int(sum(len(x) == 0 for x in df["sufficiency_tokens"]))


# -----------------------------
# SAVE OUTPUT
# -----------------------------

output_path = os.path.join(BASE_DIR, "data", "train_eval_variants_step8.csv")
df.to_csv(output_path, index=False)


# -----------------------------
# PRINT SUMMARY
# -----------------------------

print("\nEVALUATION VARIANTS CREATED SUCCESSFULLY")
print("Columns added:")
print("- sufficiency_tokens")
print("- comprehensiveness_tokens")
print(f"Average sufficiency length       : {avg_suff_len:.2f}")
print(f"Average comprehensiveness length : {avg_comp_len:.2f}")
print(f"Samples with no concept tokens   : {samples_no_concepts}")


# -----------------------------
# CENTRAL LOG UPDATE
# -----------------------------

update_log("Step8_EvaluationVariants", {
    "dataset_name": "LIAR",
    "total_samples": len(df),
    "avg_sufficiency_length": round(avg_suff_len, 2),
    "avg_comprehensiveness_length": round(avg_comp_len, 2),
    "samples_with_no_concepts": samples_no_concepts,
    "evaluation_type": "Sufficiency & Comprehensiveness"
})
