import pandas as pd
import numpy as np
from scipy.signal import stft
import matplotlib.pyplot as plt
import os

def generate_spectrogram():
    df = pd.read_csv("data/processed/processed_data.csv", index_col=0)

    signal = df.iloc[:, 0].values  # use first stock

    f, t, Zxx = stft(signal, nperseg=32)
    spectrogram = np.abs(Zxx)

    os.makedirs("outputs/plots", exist_ok=True)

    plt.pcolormesh(t, f, spectrogram)
    plt.title("Spectrogram")
    plt.ylabel("Frequency")
    plt.xlabel("Time")
    plt.savefig("outputs/plots/spectrogram.png")
    plt.close()

    np.save("data/processed/spectrogram.npy", spectrogram)

    print("Spectrogram generated")

if __name__ == "__main__":
    generate_spectrogram()