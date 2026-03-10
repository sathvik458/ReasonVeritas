import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy as np
import json
from sklearn.utils.class_weight import compute_class_weight
import csv
import os
os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)

from torch.utils.data import TensorDataset, DataLoader
from bilstm_attention import BiLSTMAttention

#Load the encoded dataset
with open("data/encoded_L128.pkl", "rb") as f:
    data = pickle.load(f)

X_train = torch.tensor(data["train"]["indices"])

# To convert label into binary
label_map = {"fake": 0, "real": 1}

y_train = torch.tensor(
    [label_map[l] for l in data["train"]["binary_label"]],
    dtype=torch.long
)

X_val = torch.tensor(data["val"]["indices"])
y_val = torch.tensor(
    [label_map[l] for l in data["val"]["binary_label"]],
    dtype=torch.long
)


# Convert dataset to tensor
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)


# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load model values from phase 2
# Load embedding matrix
embedding_matrix = np.load("data/embedding_matrix.npy")

# Extract dimensions
vocab_size = embedding_matrix.shape[0]
embed_dim = embedding_matrix.shape[1]

num_classes = len(torch.unique(y_train))  # To determine the no of classes

# Model defination and initialization
model = BiLSTMAttention(
    vocab_size=vocab_size,
    embed_dim=embed_dim,
    hidden_size=128,
    num_classes=num_classes
)

model = model.to(device)  # Move model to GPU


# Load GloVe embedding matrix

model.embedding = nn.Embedding.from_pretrained(
    torch.tensor(embedding_matrix, dtype=torch.float32),
    freeze=False
).to(device)


# Load class weights 
classes = np.unique(y_train.numpy())

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train.numpy()
)

class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)  # Loss function created

optimizer = optim.Adam(model.parameters(), lr=2e-4) 


# To save logs
log_file = "logs/training_log.csv"

with open(log_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss", "val_loss", "val_accuracy"])


# Training configuration

num_epochs = 50
best_val_acc = 0

for epoch in range(num_epochs):

    model.train()
    total_loss = 0

    for input_ids, labels in train_loader:

        input_ids = input_ids.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits, attention = model(input_ids)

        loss = criterion(logits, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    
    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_loss:.4f}")


# Validation

    model.eval()

    correct = 0
    total = 0
    val_loss = 0

    with torch.no_grad():

        for input_ids, labels in val_loader:

            input_ids = input_ids.to(device)
            labels = labels.to(device)

            logits, _ = model(input_ids)

            loss = criterion(logits, labels)
            val_loss += loss.item()

            preds = torch.argmax(logits, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_loss = val_loss / len(val_loader)
    accuracy = correct / total
    
    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch+1, avg_loss, val_loss, accuracy])

    if accuracy > best_val_acc:
        best_val_acc = accuracy
        torch.save(model.state_dict(), "models/bilstm_attention_best.pt")
        print("Model saved!")

    print(f"Validation Loss: {val_loss:.4f} | Accuracy: {accuracy:.4f}")