import pandas as pd
import os
import sys
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _src)
from utils_logger import update_log
from config import DATA_DIR, MAX_SEQUENCE_LEN


# -----------------------------
# LOAD DATA (AFTER STEP-4)
# -----------------------------

input_path = os.path.join(DATA_DIR, f"liar_truncated_L{MAX_SEQUENCE_LEN}.csv")
df = pd.read_csv(input_path)


# -----------------------------
# MULTI-CLASS DISTRIBUTION
# -----------------------------

class_counts = df["label"].value_counts()
total_samples = len(df)

print("\nCLASS DISTRIBUTION")
print(class_counts)

print("\nClass Percentages:")
for label, count in class_counts.items():
    percent = (count / total_samples) * 100
    print(f"{label}: {percent:.2f}%")


# -----------------------------
# BINARY CONVERSION
# -----------------------------

fake_labels = ["false", "barely-true", "pants-fire"]
real_labels = ["half-true", "mostly-true", "true"]

df["binary_label"] = df["label"].apply(
    lambda x: "fake" if x in fake_labels else "real"
)

binary_counts = df["binary_label"].value_counts()

print("\nBINARY CLASS DISTRIBUTION")
print(binary_counts)


# -----------------------------
# COMPUTE CLASS WEIGHTS
# -----------------------------

classes = np.array(binary_counts.index)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=df["binary_label"]
)

class_weights = dict(zip(classes, weights))

print("\nCLASS WEIGHTS (For Loss Function)")
print(class_weights)


# -----------------------------
# SAVE WEIGHTS
# -----------------------------

os.makedirs(DATA_DIR, exist_ok=True)
weights_path = os.path.join(DATA_DIR, "class_weights.txt")
with open(weights_path, "w") as f:
    for k, v in class_weights.items():
        f.write(f"{k}: {float(v)}\n")

print("\nClass weights saved successfully.")


# -----------------------------
# CENTRAL LOG UPDATE
# -----------------------------

update_log("Step5_ClassBalance", {
    "dataset_name": "LIAR",
    "total_samples": total_samples,
    "multi_class_distribution": class_counts.to_dict(),
    "binary_real_count": int(binary_counts.get("real", 0)),
    "binary_fake_count": int(binary_counts.get("fake", 0)),
    "binary_real_percentage": round((binary_counts.get("real", 0) / total_samples) * 100, 2),
    "binary_fake_percentage": round((binary_counts.get("fake", 0) / total_samples) * 100, 2),
    "class_weights": {
        str(k): float(v) for k, v in class_weights.items()
    }
})
