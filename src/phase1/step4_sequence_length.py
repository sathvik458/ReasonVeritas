import pandas as pd
import os
import ast
import sys

_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _src)
from utils_logger import update_log
from config import DATA_DIR, SEQUENCE_LENGTHS


# -----------------------------
# LOAD TOKENIZED DATA (STEP-3)
# -----------------------------

input_path = os.path.join(DATA_DIR, "liar_tokenized_step3.csv")
df = pd.read_csv(input_path)
df["tokens"] = df["tokens"].apply(ast.literal_eval)

original_avg_length = sum(len(t) for t in df["tokens"]) / len(df)


# -----------------------------
# HEAD-FOCUSED TRUNCATION
# -----------------------------

def truncate_sequence(tokens, max_len):
    tokens = list(tokens)
    if len(tokens) > max_len:
        return tokens[:max_len]
    return tokens


# -----------------------------
# PRODUCE ONE FILE PER L (128, 256, 512)
# -----------------------------

os.makedirs(DATA_DIR, exist_ok=True)
summaries = []

for L in SEQUENCE_LENGTHS:
    df_out = df.copy()
    df_out["truncated_tokens"] = df_out["tokens"].apply(lambda t: truncate_sequence(t, L))
    num_truncated = sum(len(t) > L for t in df["tokens"])
    truncated_avg = sum(len(t) for t in df_out["truncated_tokens"]) / len(df_out)
    output_path = os.path.join(DATA_DIR, f"liar_truncated_L{L}.csv")
    df_out.to_csv(output_path, index=False)
    summaries.append((L, truncated_avg, num_truncated))
    update_log(f"Step4_SequenceLengthHandling_L{L}", {
        "dataset_name": "LIAR",
        "total_samples": len(df_out),
        "max_sequence_length": L,
        "original_avg_length": round(original_avg_length, 2),
        "truncated_avg_length": round(truncated_avg, 2),
        "num_truncated_samples": num_truncated
    })


# -----------------------------
# PRINT SUMMARY
# -----------------------------

print("\nSEQUENCE LENGTH HANDLING SUMMARY (L = 128, 256, 512)")
print(f"Original avg length           : {original_avg_length:.2f}")
for L, truncated_avg, num_truncated in summaries:
    print(f"  L={L}: truncated avg = {truncated_avg:.2f}, samples truncated = {num_truncated}")
print(f"Outputs: liar_truncated_L128.csv, liar_truncated_L256.csv, liar_truncated_L512.csv")
