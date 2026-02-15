import pandas as pd
import spacy
import os
import sys

_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _src)
from utils_logger import update_log
from config import DATA_DIR


# -----------------------------
# LOAD NORMALIZED DATA (STEP-2)
# -----------------------------

input_path = os.path.join(DATA_DIR, "liar_normalized_step2.csv")
df = pd.read_csv(input_path)


# -----------------------------
# LOAD SPACY MODEL
# -----------------------------

# Disable parser and NER for faster tokenization
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])


# -----------------------------
# LOGGING
# -----------------------------

log = {
    "total_tokens": 0,
    "punctuation_tokens": 0,
    "negation_tokens": 0
}

NEGATIONS = {"not", "no", "never", "none", "n't"}


# -----------------------------
# TOKENIZATION FUNCTION
# -----------------------------

def tokenize_text(text):

    text = str(text)  # Safety cast
    doc = nlp(text)

    tokens = []

    for token in doc:
        tokens.append(token.text)

        # Count punctuation
        if token.is_punct:
            log["punctuation_tokens"] += 1

        # Count negation words
        if token.text.lower() in NEGATIONS:
            log["negation_tokens"] += 1

    log["total_tokens"] += len(tokens)

    return tokens


# -----------------------------
# APPLY TOKENIZATION
# -----------------------------

df["tokens"] = df["normalized_text"].apply(tokenize_text)

avg_tokens = log["total_tokens"] / len(df)


# -----------------------------
# SAVE OUTPUT
# -----------------------------

os.makedirs(DATA_DIR, exist_ok=True)
output_path = os.path.join(DATA_DIR, "liar_tokenized_step3.csv")
df.to_csv(output_path, index=False)


# -----------------------------
# PRINT SUMMARY
# -----------------------------

print("\nTOKENIZATION SUMMARY")
print(f"Total tokens                 : {log['total_tokens']}")
print(f"Average tokens per sentence  : {avg_tokens:.2f}")
print(f"Punctuation tokens detected  : {log['punctuation_tokens']}")
print(f"Negation tokens detected     : {log['negation_tokens']}")


# -----------------------------
# CENTRAL LOG UPDATE
# -----------------------------

update_log("Step3_Tokenization", {
    "dataset_name": "LIAR",
    "total_samples": len(df),
    "total_tokens": log["total_tokens"],
    "avg_tokens_per_sentence": round(avg_tokens, 2),
    "punctuation_tokens": log["punctuation_tokens"],
    "negation_tokens": log["negation_tokens"]
})
