import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("logs/training_log.csv")

plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
plt.plot(df["epoch"], df["val_loss"], label="Validation Loss")

plt.legend()
plt.show()