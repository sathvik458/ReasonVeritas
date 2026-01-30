import pandas as pd
import re
import unicodedata
import emoji
import contractions
from bs4 import BeautifulSoup
import os



# LOAD DATA


# Get the main project directory (one level above src folder)
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# Create full paths to dataset files inside the "data" folder
train_path = os.path.join(BASE_DIR, "data", "train.tsv")
valid_path = os.path.join(BASE_DIR, "data", "valid.tsv")
test_path  = os.path.join(BASE_DIR, "data", "test.tsv")

# LIAR dataset does not contain column headers,
# so we use header=None to load it correctly
train = pd.read_csv(train_path, sep="\t", header=None)
valid = pd.read_csv(valid_path, sep="\t", header=None)
test  = pd.read_csv(test_path,  sep="\t", header=None)

# Combine train, validation and test into one dataframe
df = pd.concat([train, valid, test], ignore_index=True)

# Column index 2 contains the actual statement
# Column index 1 contains the label
df = df[[2, 1]]
df.columns = ["statement", "label"]

# Remove rows where statement or label is missing
df = df.dropna()



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
output_path = os.path.join(BASE_DIR, "data", "liar_cleaned_step1.csv")
df.to_csv(output_path, index=False)



# PRINT CLEANING SUMMARY


print("\nCLEANING SUMMARY")
print(f"URLs replaced       : {log['urls']}")
print(f"Emojis mapped       : {log['emojis']}")
print(f"Negations retained  : {log['negations']}")
print(f"Avg len before      : {sum(log['len_before'])/len(log['len_before']):.2f}")
print(f"Avg len after       : {sum(log['len_after'])/len(log['len_after']):.2f}")
