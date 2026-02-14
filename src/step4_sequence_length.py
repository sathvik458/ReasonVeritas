import pandas as pd
import os
import ast
from utils_logger import update_log


# -----------------------------
# LOAD TOKENIZED DATA (STEP-3)
# -----------------------------

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
input_path = os.path.join(BASE_DIR, "data", "liar_tokenized_step3.csv")

df = pd.read_csv(input_path)

# Convert string representation of list back to Python list
df["tokens"] = df["tokens"].apply(ast.literal_eval)


# -----------------------------
# CONFIGURE MAX SEQUENCE LENGTH
# -----------------------------

MAX_LEN = 32   # You can test 32, 64, 128


# -----------------------------
# LOGGING
# -----------------------------

log = {
    "num_truncated_samples": 0
}


# -----------------------------
# HEAD-FOCUSED TRUNCATION
# -----------------------------

def truncate_sequence(tokens):

    tokens = list(tokens)  # safety

    if len(tokens) > MAX_LEN:
        log["num_truncated_samples"] += 1
        return tokens[:MAX_LEN]   # Keep first L tokens
    else:
        return tokens


# -----------------------------
# COMPUTE ORIGINAL AVG LENGTH
# -----------------------------

original_avg_length = sum(len(t) for t in df["tokens"]) / len(df)


# -----------------------------
# APPLY TRUNCATION
# -----------------------------

df["truncated_tokens"] = df["tokens"].apply(truncate_sequence)


# -----------------------------
# COMPUTE TRUNCATED AVG LENGTH
# -----------------------------

truncated_avg_length = sum(len(t) for t in df["truncated_tokens"]) / len(df)


# -----------------------------
# SAVE OUTPUT
# -----------------------------

output_path = os.path.join(BASE_DIR, "data", f"liar_truncated_L{MAX_LEN}.csv")
df.to_csv(output_path, index=False)


# -----------------------------
# PRINT SUMMARY
# -----------------------------

print("\nSEQUENCE LENGTH HANDLING SUMMARY")
print(f"Max sequence length (L)       : {MAX_LEN}")
print(f"Original avg length           : {original_avg_length:.2f}")
print(f"Truncated avg length          : {truncated_avg_length:.2f}")
print(f"Number of truncated samples   : {log['num_truncated_samples']}")


# -----------------------------
# CENTRAL LOG UPDATE
# -----------------------------

update_log("Step4_SequenceLengthHandling", {
    "dataset_name": "LIAR",
    "total_samples": len(df),
    "max_sequence_length": MAX_LEN,
    "original_avg_length": round(original_avg_length, 2),
    "truncated_avg_length": round(truncated_avg_length, 2),
    "num_truncated_samples": log["num_truncated_samples"]
})
