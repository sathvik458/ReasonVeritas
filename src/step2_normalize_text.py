import pandas as pd
import re
import unicodedata
import contractions
import os


# LOAD STEP-1 OUTPUT


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
input_path = os.path.join(BASE_DIR, "data", "liar_cleaned_step1.csv")

df = pd.read_csv(input_path)


# LOGGING DICTIONARY


log = {
    "len_before": [],
    "len_after": [],
    "contractions_expanded": 0,
    "allcaps_detected": 0,
    "numbers_normalized": 0
}


# NORMALIZATION FUNCTION


def normalize_text(text):

    # Length before normalization
    log["len_before"].append(len(text.split()))

    # Unicode normalization (safety)
    text = unicodedata.normalize("NFKC", text)

    # Count ALLCAPS tokens (already tagged in Step-1)
    allcaps_count = len(re.findall(r"<allcaps>", text))
    log["allcaps_detected"] += allcaps_count

    # Expand contractions again (if any slipped)
    expanded_text = contractions.fix(text)
    if expanded_text != text:
        log["contractions_expanded"] += 1
    text = expanded_text

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Normalize numbers (already <NUM>, but count)
    numbers = re.findall(r"<num>", text)
    log["numbers_normalized"] += len(numbers)

    # Standard punctuation normalization
    text = re.sub(r"([.,!?]){2,}", r"\1", text)

    # Length after normalization
    log["len_after"].append(len(text.split()))

    return text



# APPLY NORMALIZATION


df["normalized_text"] = df["clean_statement"].apply(normalize_text)

# SAVE STEP-2 OUTPUT


output_path = os.path.join(BASE_DIR, "data", "liar_normalized_step2.csv")
df.to_csv(output_path, index=False)


# PRINT NORMALIZATION SUMMARY


print("\nNORMALIZATION SUMMARY")
print(f"Contractions expanded    : {log['contractions_expanded']}")
print(f"ALLCAPS detected         : {log['allcaps_detected']}")
print(f"Numbers normalized       : {log['numbers_normalized']}")
print(f"Avg len before           : {sum(log['len_before'])/len(log['len_before']):.2f}")
print(f"Avg len after            : {sum(log['len_after'])/len(log['len_after']):.2f}")
