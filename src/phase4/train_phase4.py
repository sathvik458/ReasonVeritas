"""
phase4/train_phase4.py

Training script for the Chain-of-Thought model (Phase 4).

Key differences from Phase 3's train_model.py:
  1. Loads concept masks alongside token indices
  2. Uses CoTModel instead of BiLSTMAttention
  3. Computes FOUR losses per batch:
       - main_loss     : CrossEntropy on final_logits vs true label
       - aux_loss_e    : CrossEntropy on emotion head's logits vs true label
       - aux_loss_m    : CrossEntropy on modality head's logits vs true label
       - aux_loss_n    : CrossEntropy on negation head's logits vs true label
  4. Total loss = main_loss + lambda * (aux_e + aux_m + aux_n)
  5. Saves rationale examples at end of each epoch

Run from project root:
    python src/phase4/train_phase4.py
"""

import os
import sys
import pickle
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight

# ---- Path setup ----
# ---- Path setup (FINAL FIX) ----
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))        # src/phase4/
SRC_DIR = os.path.dirname(CURRENT_DIR)                          # src/
PROJECT_ROOT = os.path.dirname(SRC_DIR)                         # ReasonVeritas/

# Add BOTH src and project root
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, PROJECT_ROOT)



from config import DATA_DIR, EMBEDDING_DIM
from phase4.cot_model import CoTModel
from phase4.concept_masks import load_masks_for_split, masks_to_batch_tensor
from phase4.rationale import generate_batch_rationales

os.makedirs("logs",   exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs(os.path.join("logs", "rationales"), exist_ok=True)

# -----------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------
L           = 128          # Sequence length — change to 256 or 512 to ablate
BATCH_SIZE  = 32
NUM_EPOCHS  = 50
LR          = 2e-4
LAMBDA_AUX  = 0.3          # Weight for auxiliary losses
                            # Total loss = main + 0.3*(aux_e + aux_m + aux_n)
                            # Increase to force stronger concept reasoning
                            # Decrease if val accuracy drops below Phase 3 baseline

HIDDEN_SIZE = 128

label_map     = {"fake": 0, "real": 1}
label_map_inv = {0: "FAKE", 1: "REAL"}

# -----------------------------------------------------------------------
# LOAD ENCODED DATA (Phase 2 output)
# -----------------------------------------------------------------------
print(f"Loading encoded data for L={L}...")
with open(os.path.join(DATA_DIR, f"encoded_L{L}.pkl"), "rb") as f:
    data = pickle.load(f)

X_train = torch.tensor(data["train"]["indices"])
y_train = torch.tensor(
    [label_map[l] for l in data["train"]["binary_label"]], dtype=torch.long
)

X_val = torch.tensor(data["val"]["indices"])
y_val = torch.tensor(
    [label_map[l] for l in data["val"]["binary_label"]], dtype=torch.long
)

print(f"  Train: {len(X_train)} samples | Val: {len(X_val)} samples")

# -----------------------------------------------------------------------
# LOAD CONCEPT MASKS (Phase 1 step 6 output, via Phase 4 concept_masks.py)
# The split CSVs (train_split_L128.csv etc.) contain concept_tags because
# step7 splits liar_concepts_step6_L128.csv which already has that column.
# -----------------------------------------------------------------------
print("Loading concept masks...")
train_csv = os.path.join(DATA_DIR, f"train_split_L{L}.csv")
val_csv   = os.path.join(DATA_DIR, f"val_split_L{L}.csv")

train_e_masks, train_m_masks, train_n_masks = load_masks_for_split(train_csv, L)
val_e_masks,   val_m_masks,   val_n_masks   = load_masks_for_split(val_csv,   L)

# Convert mask lists to tensors (shape: N × L)
train_E = torch.tensor(train_e_masks, dtype=torch.float32)
train_M = torch.tensor(train_m_masks, dtype=torch.float32)
train_N = torch.tensor(train_n_masks, dtype=torch.float32)

val_E   = torch.tensor(val_e_masks,   dtype=torch.float32)
val_M   = torch.tensor(val_m_masks,   dtype=torch.float32)
val_N   = torch.tensor(val_n_masks,   dtype=torch.float32)

print(f"  Masks loaded. Train emotion tokens: {train_E.sum().int()}")

# -----------------------------------------------------------------------
# BUILD DATASETS AND DATALOADERS
# Each sample: (input_ids, label, emotion_mask, modality_mask, negation_mask)
# -----------------------------------------------------------------------
train_dataset = TensorDataset(X_train, y_train, train_E, train_M, train_N)
val_dataset   = TensorDataset(X_val,   y_val,   val_E,   val_M,   val_N)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE)

# -----------------------------------------------------------------------
# DEVICE
# -----------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")

# -----------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------
embedding_matrix = np.load(os.path.join(DATA_DIR, "embedding_matrix.npy"))
vocab_size = embedding_matrix.shape[0]
embed_dim  = embedding_matrix.shape[1]

model = CoTModel(
    vocab_size=vocab_size,
    embed_dim=embed_dim,
    hidden_size=HIDDEN_SIZE,
    num_classes=2
)

# Load Phase 3 GloVe embeddings into the backbone
model.backbone.embedding = nn.Embedding.from_pretrained(
    torch.tensor(embedding_matrix, dtype=torch.float32),
    freeze=False
)

model = model.to(device)
print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# -----------------------------------------------------------------------
# LOSS AND OPTIMIZER
# -----------------------------------------------------------------------
classes = np.unique(y_train.numpy())
weights = compute_class_weight("balanced", classes=classes, y=y_train.numpy())
class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

# All four losses use the same class weights — fake/real imbalance applies
# equally to the main classifier and each concept head
criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = optim.Adam(model.parameters(), lr=LR)

# -----------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------
log_file = "logs/phase4_training_log.csv"
with open(log_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "epoch", "train_loss", "main_loss", "aux_loss",
        "val_loss", "val_accuracy"
    ])

# -----------------------------------------------------------------------
# TRAINING LOOP
# -----------------------------------------------------------------------
print(f"\nStarting Phase 4 training — {NUM_EPOCHS} epochs, lambda_aux={LAMBDA_AUX}")
print("=" * 60)

best_val_acc = 0.0

for epoch in range(NUM_EPOCHS):

    # ---- TRAIN ----
    model.train()
    total_loss = 0.0
    total_main = 0.0
    total_aux  = 0.0

    for batch in train_loader:
        input_ids, labels, e_mask, m_mask, n_mask = batch
        input_ids = input_ids.to(device)
        labels    = labels.to(device)
        e_mask    = e_mask.to(device)
        m_mask    = m_mask.to(device)
        n_mask    = n_mask.to(device)

        optimizer.zero_grad()

        # Forward pass — returns 8 values
        (final_logits,
         emotion_score, modality_score, negation_score,
         aux_logits_e, aux_logits_m, aux_logits_n,
         base_alpha) = model(input_ids, e_mask, m_mask, n_mask)

        # ---- Compute losses ----
        main_loss  = criterion(final_logits, labels)

        # Each concept head independently tries to predict the label
        # This forces the head to learn concept-relevant features
        aux_loss_e = criterion(aux_logits_e, labels)
        aux_loss_m = criterion(aux_logits_m, labels)
        aux_loss_n = criterion(aux_logits_n, labels)

        aux_loss   = aux_loss_e + aux_loss_m + aux_loss_n

        # Combined loss: main prediction + weighted concept reasoning losses
        loss = main_loss + LAMBDA_AUX * aux_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_main += main_loss.item()
        total_aux  += aux_loss.item()

    avg_loss = total_loss / len(train_loader)
    avg_main = total_main / len(train_loader)
    avg_aux  = total_aux  / len(train_loader)

    # ---- VALIDATE ----
    model.eval()
    correct   = 0
    total     = 0
    val_loss  = 0.0

    # For saving example rationales
    sample_rationales = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids, labels, e_mask, m_mask, n_mask = batch
            input_ids = input_ids.to(device)
            labels    = labels.to(device)
            e_mask    = e_mask.to(device)
            m_mask    = m_mask.to(device)
            n_mask    = n_mask.to(device)

            (final_logits,
             emotion_score, modality_score, negation_score,
             aux_logits_e, aux_logits_m, aux_logits_n,
             base_alpha) = model(input_ids, e_mask, m_mask, n_mask)

            loss = criterion(final_logits, labels)
            val_loss += loss.item()

            preds   = torch.argmax(final_logits, dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

            # Save first batch's rationales for inspection
            if len(sample_rationales) == 0:
                batch_rationales = generate_batch_rationales(
                    emotion_score, modality_score, negation_score,
                    final_logits, label_map_inv
                )
                sample_rationales = batch_rationales[:3]  # save 3 examples

    val_loss = val_loss / len(val_loader)
    accuracy = correct / total

    # ---- LOG ----
    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch+1, avg_loss, avg_main, avg_aux, val_loss, accuracy])

    # ---- SAVE BEST MODEL ----
    if accuracy > best_val_acc:
        best_val_acc = accuracy
        torch.save(model.state_dict(), "models/cot_model_best.pt")
        print(f"  ✓ Model saved (val_acc={accuracy:.4f})")

    # ---- PRINT EPOCH SUMMARY ----
    print(
        f"Epoch {epoch+1:>2}/{NUM_EPOCHS} | "
        f"Loss: {avg_loss:.4f} (main={avg_main:.4f} aux={avg_aux:.4f}) | "
        f"Val Loss: {val_loss:.4f} | Val Acc: {accuracy:.4f}"
    )

    # ---- SAVE RATIONALE EXAMPLES every 5 epochs ----
    if (epoch + 1) % 5 == 0 and sample_rationales:
        rationale_path = f"logs/rationales/epoch_{epoch+1:02d}.txt"
        with open(rationale_path, "w", encoding="utf-8") as f:
            f.write(f"=== Epoch {epoch+1} — Sample Rationales ===\n\n")
            for i, (r, v, c) in enumerate(sample_rationales):
                f.write(f"--- Example {i+1} ---\n")
                f.write(r + "\n\n")
        print(f"  Rationales saved → {rationale_path}")

print("\n" + "=" * 60)
print(f"Training complete. Best val accuracy: {best_val_acc:.4f}")
print(f"Model saved to: models/cot_model_best.pt")
print(f"Training log:   logs/phase4_training_log.csv")
print(f"Rationales:     logs/rationales/")