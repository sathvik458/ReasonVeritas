import pandas as pd
import os
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


# LOAD DATA (AFTER STEP-4)


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
input_path = os.path.join(BASE_DIR, "data", "liar_truncated_L128.csv")

df = pd.read_csv(input_path)


# CHECK CLASS DISTRIBUTION


class_counts = df["label"].value_counts()
total_samples = len(df)

print("\nCLASS DISTRIBUTION")
print(class_counts)

print("\nClass Percentages:")
for label, count in class_counts.items():
    percent = (count / total_samples) * 100
    print(f"{label}: {percent:.2f}%")


# BINARY CONVERSION (OPTIONAL)


# If you want Fake vs Real binary:
# Fake = {false, barely-true, pants-fire}
# Real = {half-true, mostly-true, true}

fake_labels = ["false", "barely-true", "pants-fire"]
real_labels = ["half-true", "mostly-true", "true"]

df["binary_label"] = df["label"].apply(
    lambda x: "fake" if x in fake_labels else "real"
)

binary_counts = df["binary_label"].value_counts()

print("\nBINARY CLASS DISTRIBUTION")
print(binary_counts)


# COMPUTE CLASS WEIGHTS


classes = np.array(binary_counts.index)
weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=df["binary_label"]
)

class_weights = dict(zip(classes, weights))

print("\nCLASS WEIGHTS (For Loss Function)")
print(class_weights)


# SAVE WEIGHTS FOR TRAINING


weights_path = os.path.join(BASE_DIR, "data", "class_weights.txt")
with open(weights_path, "w") as f:
    for k, v in class_weights.items():
        f.write(f"{k}: {v}\n")

print("\nClass weights saved successfully.")
