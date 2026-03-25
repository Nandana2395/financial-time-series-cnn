import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import save_model
from model import build_model
import os

def train_model():
    spectrogram = np.load("data/processed/spectrogram.npy")

    X = []
    y = []

    signal_length = spectrogram.shape[1]
    window = 10

    for i in range(signal_length - window):
        X.append(spectrogram[:, i:i+window])
        y.append(spectrogram[0, i+window])

    X = np.array(X)
    y = np.array(y)

    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = build_model(X.shape[1:])

    model.fit(X_train, y_train, epochs=5)

    os.makedirs("outputs/models", exist_ok=True)
    model.save("outputs/models/model.h5")

    print("Model trained and saved")

if __name__ == "__main__":
    train_model()