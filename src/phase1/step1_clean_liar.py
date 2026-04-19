import pandas as pd
import re
import unicodedata
import emoji
import contractions
from bs4 import BeautifulSoup
import os
import sys

_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _src)
from config import LIAR_DATASET_DIR, DATA_DIR

# LOAD DATA

# Raw dataset from config path; outputs go to project data folder
train_path = os.path.join(LIAR_DATASET_DIR, "train.tsv")
valid_path = os.path.join(LIAR_DATASET_DIR, "valid.tsv")
test_path  = os.path.join(LIAR_DATASET_DIR, "test.tsv")

# LIAR dataset does not contain column headers,
# so we use header=None to load it correctly
train = pd.read_csv(train_path, sep="\t", header=None)
valid = pd.read_csv(valid_path, sep="\t", header=None)
test  = pd.read_csv(test_path,  sep="\t", header=None)

# Combine train, validation and test into one dataframe
df = pd.concat([train, valid, test], ignore_index=True)

# LIAR raw columns:
#   0 = id, 1 = label, 2 = statement, 3 = subject(s),
#   4 = speaker, 5 = job, 6 = state, 7 = party,
#   8 = barely_true_count, 9 = false_count, 10 = half_true_count,
#   11 = mostly_true_count, 12 = pants_fire_count, 13 = context
# We keep statement + label (required) and the metadata columns that push
# LIAR baselines from ~62 % to ~70-77 % (Wang 2017, Karimi & Tang 2019).
META_COLS = {
    1:  "label",
    2:  "statement",
    3:  "subject",
    4:  "speaker",
    7:  "party",
    8:  "barely_true_count",
    9:  "false_count",
    10: "half_true_count",
    11: "mostly_true_count",
    12: "pants_fire_count",
    13: "context",
}
df = df[list(META_COLS.keys())].rename(columns=META_COLS)

# Drop only rows missing the two required fields; metadata may legitimately be NaN
df = df.dropna(subset=["statement", "label"])

# Fill missing categorical metadata with explicit "unknown" sentinel
for col in ["subject", "speaker", "party", "context"]:
    df[col] = df[col].fillna("unknown").astype(str)

# Fill missing credit-history counts with 0 (treat as no prior record)
for col in ["barely_true_count", "false_count", "half_true_count",
            "mostly_true_count", "pants_fire_count"]:
    df[col] = df[col].fillna(0).astype(float)



# LOGGING FOR CLEANING STATS


# This dictionary will store statistics during cleaning
log = {
    "urls": 0,
    "emojis": 0,
    "negations": 0,
    "len_before": [],
    "len_after": []
}

# Set of negation words that we want to preserve
NEGATIONS = {"not", "no", "never", "none", "n't"}



# TEXT CLEANING FUNCTION


def clean_text(text):

    # Store original sentence length (before cleaning)
    log["len_before"].append(len(text.split()))

    # Remove any HTML tags if present
    text = BeautifulSoup(text, "html.parser").get_text()

    # Normalize unicode characters (fix encoding issues)
    text = unicodedata.normalize("NFKC", text)

    # Expand contractions (e.g., "can't" -> "can not")
    text = contractions.fix(text)

    # Find and count URLs
    urls = re.findall(r"http\S+|www\S+", text)
    log["urls"] += len(urls)

    # Replace URLs with placeholder instead of deleting
    text = re.sub(r"http\S+|www\S+", "<URL>", text)

    # Count emojis if present
    emoji_count = sum(c in emoji.EMOJI_DATA for c in text)
    log["emojis"] += emoji_count

    # Convert emojis to readable text form
    text = emoji.demojize(text, delimiters=(" <EMO_", "> "))

    # Mark words written fully in uppercase (for emphasis detection)
    text = re.sub(r"\b([A-Z]{2,})\b", r"<ALLCAPS> \1", text)

    # Replace numbers with a generic token
    text = re.sub(r"\b\d+\b", "<NUM>", text)

    # Reduce repeated punctuation like "!!!" -> "!"
    text = re.sub(r"([!?]){2,}", r"\1", text)

    # Convert everything to lowercase (after marking ALLCAPS)
    text = text.lower()

    # Split into tokens to count negations
    tokens = text.split()

    # Count how many negation words are still present
    log["negations"] += sum(t in NEGATIONS for t in tokens)

    # Store sentence length after cleaning
    log["len_after"].append(len(tokens))

    return text



# APPLY CLEANING TO DATASET


# Apply the cleaning function to each statement
df["clean_statement"] = df["statement"].apply(clean_text)

# Remove duplicate cleaned statements
df = df.drop_duplicates(subset="clean_statement")



# SAVE CLEANED DATA


# Save cleaned dataset inside data folder
os.makedirs(DATA_DIR, exist_ok=True)
output_path = os.path.join(DATA_DIR, "liar_cleaned_step1.csv")
df.to_csv(output_path, index=False)



# PRINT CLEANING SUMMARY


print("\nCLEANING SUMMARY")
print(f"URLs replaced       : {log['urls']}")
print(f"Emojis mapped       : {log['emojis']}")
print(f"Negations retained  : {log['negations']}")
print(f"Avg len before      : {sum(log['len_before'])/len(log['len_before']):.2f}")
print(f"Avg len after       : {sum(log['len_after'])/len(log['len_after']):.2f}")


from utils_logger import update_log

original_count = len(train) + len(valid) + len(test)
cleaned_count = len(df)

update_log("Step1_Cleaning", {
    "dataset_name": "LIAR",
    "total_samples": cleaned_count,
    "duplicates_removed": original_count - cleaned_count,
    "urls_replaced": log["urls"],
    "emojis_mapped": log["emojis"],
    "negations_retained": log["negations"],
    "avg_len_before": round(sum(log["len_before"]) / len(log["len_before"]), 2),
    "avg_len_after": round(sum(log["len_after"]) / len(log["len_after"]), 2)
})
