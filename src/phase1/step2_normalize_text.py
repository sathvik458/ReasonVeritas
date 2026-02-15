import pandas as pd
import re
import unicodedata
import contractions
import os
import sys

_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _src)
from utils_logger import update_log
from config import DATA_DIR


# -----------------------------
# LOAD STEP-1 OUTPUT
# -----------------------------

input_path = os.path.join(DATA_DIR, "liar_cleaned_step1.csv")
df = pd.read_csv(input_path)


# -----------------------------
# LOGGING DICTIONARY
# -----------------------------

log = {
    "len_before": [],
    "len_after": [],
    "allcaps_detected": 0,
    "numbers_normalized": 0
}


# -----------------------------
# NORMALIZATION FUNCTION
# -----------------------------

def normalize_text(text):

    # Length before normalization
    log["len_before"].append(len(str(text).split()))

    # Unicode normalization
    text = unicodedata.normalize("NFKC", str(text))

    # Count ALLCAPS tokens (from Step-1)
    allcaps_count = len(re.findall(r"<allcaps>", text))
    log["allcaps_detected"] += allcaps_count

    # Expand contractions
    text = contractions.fix(text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Count normalized numbers (from Step-1)
    numbers = re.findall(r"<num>", text)
    log["numbers_normalized"] += len(numbers)

    # Standard punctuation normalization
    text = re.sub(r"([.,!?]){2,}", r"\1", text)

    # Length after normalization
    log["len_after"].append(len(text.split()))

    return text


# -----------------------------
# APPLY NORMALIZATION
# -----------------------------

df["normalized_text"] = df["clean_statement"].apply(normalize_text)


# -----------------------------
# SAVE STEP-2 OUTPUT
# -----------------------------

os.makedirs(DATA_DIR, exist_ok=True)
output_path = os.path.join(DATA_DIR, "liar_normalized_step2.csv")
df.to_csv(output_path, index=False)


# -----------------------------
# CALCULATE AVERAGES SAFELY
# -----------------------------

avg_len_before = sum(log["len_before"]) / len(log["len_before"])
avg_len_after = sum(log["len_after"]) / len(log["len_after"])


# -----------------------------
# PRINT SUMMARY
# -----------------------------

print("\nNORMALIZATION SUMMARY")
print(f"ALLCAPS detected         : {log['allcaps_detected']}")
print(f"Numbers normalized       : {log['numbers_normalized']}")
print(f"Avg len before           : {avg_len_before:.2f}")
print(f"Avg len after            : {avg_len_after:.2f}")


# -----------------------------
# CENTRAL LOG UPDATE
# -----------------------------

update_log("Step2_Normalization", {
    "dataset_name": "LIAR",
    "total_samples": len(df),
    "allcaps_detected": log["allcaps_detected"],
    "numbers_normalized": log["numbers_normalized"],
    "avg_len_before": round(avg_len_before, 2),
    "avg_len_after": round(avg_len_after, 2)
})
