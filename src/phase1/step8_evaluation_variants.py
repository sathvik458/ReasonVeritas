import pandas as pd
import os
import ast
import sys

_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _src)
from utils_logger import update_log
from config import DATA_DIR, SEQUENCE_LENGTHS


def keep_concepts(tokens, concepts):
    return [token for token, concept in zip(tokens, concepts) if concept != "O"]


def remove_concepts(tokens, concepts):
    return [token for token, concept in zip(tokens, concepts) if concept == "O"]


# -----------------------------
# EVALUATION VARIANTS FOR EACH L (128, 256, 512)
# -----------------------------

os.makedirs(DATA_DIR, exist_ok=True)

for L in SEQUENCE_LENGTHS:
    input_path = os.path.join(DATA_DIR, f"train_split_L{L}.csv")
    df = pd.read_csv(input_path)
    df["truncated_tokens"] = df["truncated_tokens"].apply(ast.literal_eval)
    df["concept_tags"] = df["concept_tags"].apply(ast.literal_eval)

    df["sufficiency_tokens"] = df.apply(
        lambda row: keep_concepts(row["truncated_tokens"], row["concept_tags"]),
        axis=1
    )
    df["comprehensiveness_tokens"] = df.apply(
        lambda row: remove_concepts(row["truncated_tokens"], row["concept_tags"]),
        axis=1
    )

    avg_suff_len = df["sufficiency_tokens"].apply(len).mean()
    avg_comp_len = df["comprehensiveness_tokens"].apply(len).mean()
    samples_no_concepts = int(sum(len(x) == 0 for x in df["sufficiency_tokens"]))

    output_path = os.path.join(DATA_DIR, f"train_eval_variants_step8_L{L}.csv")
    df.to_csv(output_path, index=False)

    update_log(f"Step8_EvaluationVariants_L{L}", {
        "dataset_name": "LIAR",
        "total_samples": len(df),
        "avg_sufficiency_length": round(avg_suff_len, 2),
        "avg_comprehensiveness_length": round(avg_comp_len, 2),
        "samples_with_no_concepts": samples_no_concepts,
        "evaluation_type": "Sufficiency & Comprehensiveness"
    })

print("\nEVALUATION VARIANTS CREATED (L = 128, 256, 512)")
print("Outputs: train_eval_variants_step8_L128.csv, _L256.csv, _L512.csv")
