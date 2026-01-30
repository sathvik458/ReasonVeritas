import pandas as pd
import os
from sklearn.model_selection import train_test_split


# LOAD DATA (STEP-6 OUTPUT)


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
input_path = os.path.join(BASE_DIR, "data", "liar_concepts_step6.csv")

df = pd.read_csv(input_path)

# Use binary_label if created earlier
if "binary_label" not in df.columns:
    fake_labels = ["false", "barely-true", "pants-fire"]
    df["binary_label"] = df["label"].apply(
        lambda x: "fake" if x in fake_labels else "real"
    )


# STRATIFIED SPLIT (70-15-15)


# First split: Train (70%) and Temp (30%)
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["binary_label"],
    random_state=42
)

# Second split: Validation (15%) and Test (15%)
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["binary_label"],
    random_state=42
)


# SAVE SPLITS


train_path = os.path.join(BASE_DIR, "data", "train_split.csv")
val_path = os.path.join(BASE_DIR, "data", "val_split.csv")
test_path = os.path.join(BASE_DIR, "data", "test_split.csv")

train_df.to_csv(train_path, index=False)
val_df.to_csv(val_path, index=False)
test_df.to_csv(test_path, index=False)


# PRINT SUMMARY


print("\nDATASET SPLIT SUMMARY")
print(f"Train size      : {len(train_df)}")
print(f"Validation size : {len(val_df)}")
print(f"Test size       : {len(test_df)}")

print("\nTrain distribution:")
print(train_df["binary_label"].value_counts())

print("\nValidation distribution:")
print(val_df["binary_label"].value_counts())

print("\nTest distribution:")
print(test_df["binary_label"].value_counts())
