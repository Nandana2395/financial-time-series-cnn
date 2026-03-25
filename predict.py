import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

def predict():
    model = load_model("outputs/models/model.h5")
    spectrogram = np.load("data/processed/spectrogram.npy")

    X = []
    y = []

    window = 10

    for i in range(spectrogram.shape[1] - window):
        X.append(spectrogram[:, i:i+window])
        y.append(spectrogram[0, i+window])

    X = np.array(X)
    y = np.array(y)

    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

    predictions = model.predict(X)

    plt.plot(y, label="Actual")
    plt.plot(predictions, label="Predicted")
    plt.legend()
    plt.savefig("outputs/plots/prediction.png")
    plt.close()

    print("Prediction completed")

if __name__ == "__main__":
    predict()