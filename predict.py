import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error
import os

def make_predictions():
    data = np.load("data/processed/spectrogram.npy")

    # 🔥 Sliding window (same as train)
    window_size = 5
    X = []
    y = []

    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size][0])

    X = np.array(X)
    y = np.array(y)

    # Add channel dimension
    X = X.reshape((X.shape[0], X.shape[1], X.shape[2], 1))

    # Load model
    model = load_model("outputs/models/model.keras")

    # Predictions
    predictions = model.predict(X)

    # 🔥 Evaluation
    mse = mean_squared_error(y, predictions)
    rmse = np.sqrt(mse)

    print("MSE:", mse)
    print("RMSE:", rmse)

    os.makedirs("outputs/plots", exist_ok=True)

    # =========================================
    # 🔥 UPGRADE 2: Better Prediction Graph
    # =========================================
    plt.figure(figsize=(10,5))

    plt.plot(y, label="Actual", linewidth=2)
    plt.plot(predictions, label="Predicted", linestyle='--')

    plt.title("Stock Prediction Comparison")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid()

    plt.savefig("outputs/plots/prediction.png")
    plt.close()

    # =========================================
    # 🔥 UPGRADE 3: Performance Graph
    # =========================================
    plt.figure(figsize=(6,4))

    plt.bar(["MSE", "RMSE"], [mse, rmse])
    plt.title("Model Performance")

    plt.savefig("outputs/plots/performance.png")
    plt.close()

    print("Prediction completed")