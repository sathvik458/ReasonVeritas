import pandas as pd
import os
import ast


# LOAD TOKENIZED DATA (STEP-3)


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
input_path = os.path.join(BASE_DIR, "data", "liar_tokenized_step3.csv")

df = pd.read_csv(input_path)

# Convert string representation of list back to Python list
df["tokens"] = df["tokens"].apply(ast.literal_eval)


# CONFIGURE MAX SEQUENCE LENGTH


MAX_LEN = 32  # You can test 32, 64, 128


# LOGGING


log = {
    "original_avg_length": 0,
    "truncated_avg_length": 0,
    "num_truncated_samples": 0
}


# HEAD-FOCUSED TRUNCATION


def truncate_sequence(tokens):

    original_length = len(tokens)

    if original_length > MAX_LEN:
        log["num_truncated_samples"] += 1
        return tokens[:MAX_LEN]   # keep first L tokens
    else:
        return tokens

# Compute original average length
log["original_avg_length"] = sum(len(t) for t in df["tokens"]) / len(df)

# Apply truncation
df["truncated_tokens"] = df["tokens"].apply(truncate_sequence)

# Compute truncated average length
log["truncated_avg_length"] = sum(len(t) for t in df["truncated_tokens"]) / len(df)


# SAVE OUTPUT


output_path = os.path.join(BASE_DIR, "data", f"liar_truncated_L{MAX_LEN}.csv")
df.to_csv(output_path, index=False)


# PRINT SUMMARY


print("\nSEQUENCE LENGTH HANDLING SUMMARY")
print(f"Max sequence length (L)       : {MAX_LEN}")
print(f"Original avg length           : {log['original_avg_length']:.2f}")
print(f"Truncated avg length          : {log['truncated_avg_length']:.2f}")
print(f"Number of truncated samples   : {log['num_truncated_samples']}")
