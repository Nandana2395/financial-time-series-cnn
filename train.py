import numpy as np
from sklearn.model_selection import train_test_split
from src.model import build_model
import matplotlib.pyplot as plt
import os

def train_model():
    data = np.load("data/processed/spectrogram.npy")

    # 🔥 Create multiple samples (sliding window)
    window_size = 5
    X = []
    y = []

    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size][0])  # predict next value

    X = np.array(X)
    y = np.array(y)

    # Add channel dimension
    X = X.reshape((X.shape[0], X.shape[1], X.shape[2], 1))

    print("Total samples:", X.shape[0])

    # ✅ Now split works
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = build_model((X.shape[1], X.shape[2], 1))

    history = model.fit(X_train, y_train, epochs=20)

    os.makedirs("outputs/models", exist_ok=True)
    model.save("outputs/models/model.keras")

    # Plot loss
    plt.plot(history.history['loss'])
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    os.makedirs("outputs/plots", exist_ok=True)
    plt.savefig("outputs/plots/training_loss.png")
    plt.close()

    print("Model trained and saved")