import numpy as np
from scipy.signal import stft
import matplotlib.pyplot as plt
import os

def generate_spectrogram():
    data = np.load("data/processed/processed.npy")

    stock_names = ["TCS", "INFY", "RELIANCE"]

    os.makedirs("outputs/plots", exist_ok=True)

    for i in range(data.shape[1]):
        signal = data[:, i]

        f, t, Zxx = stft(signal, nperseg=32)

        spectrogram = np.abs(Zxx)

        # Normalize
        spectrogram = spectrogram / np.max(spectrogram)

        # Log scale
        spectrogram = 10 * np.log10(spectrogram + 1e-10)

        # Save spectrogram data (optional)
        np.save(f"data/processed/spectrogram_{stock_names[i]}.npy", spectrogram)

        # Plot
        plt.figure(figsize=(10, 5))
        plt.imshow(spectrogram, aspect='auto', cmap='jet')
        plt.colorbar(label="Intensity (dB)")
        plt.title(f"{stock_names[i]} Spectrogram")
        plt.xlabel("Time")
        plt.ylabel("Frequency")

        plt.savefig(f"outputs/plots/{stock_names[i]}_spectrogram.png")
        plt.close()

    print("Individual spectrograms generated")