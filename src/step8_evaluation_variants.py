import pandas as pd
import os
import ast

# LOAD SPLIT DATA (TRAIN ONLY FOR NOW)


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
input_path = os.path.join(BASE_DIR, "data", "train_split.csv")

df = pd.read_csv(input_path)

# Convert string lists back to Python lists
df["truncated_tokens"] = df["truncated_tokens"].apply(ast.literal_eval)
df["concept_tags"] = df["concept_tags"].apply(ast.literal_eval)


# SUFFICIENCY VARIANT
# Keep only concept tokens


def keep_concepts(tokens, concepts):
    return [token for token, concept in zip(tokens, concepts) if concept != "O"]


# COMPREHENSIVENESS VARIANT
# Remove concept tokens


def remove_concepts(tokens, concepts):
    return [token for token, concept in zip(tokens, concepts) if concept == "O"]

# Apply transformations
df["sufficiency_tokens"] = df.apply(
    lambda row: keep_concepts(row["truncated_tokens"], row["concept_tags"]),
    axis=1
)

df["comprehensiveness_tokens"] = df.apply(
    lambda row: remove_concepts(row["truncated_tokens"], row["concept_tags"]),
    axis=1
)


# SAVE OUTPUT


output_path = os.path.join(BASE_DIR, "data", "train_eval_variants_step8.csv")
df.to_csv(output_path, index=False)

print("\nEVALUATION VARIANTS CREATED SUCCESSFULLY")
print("Columns added:")
print("- sufficiency_tokens")
print("- comprehensiveness_tokens")
