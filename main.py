from src.data_collection import fetch_data
from src.preprocessing import preprocess
from src.stft_processing import generate_spectrogram
from src.train import train_model
from src.predict import make_predictions

def main():
    print("Step 1: Fetching Data...")
    fetch_data()

    print("Step 2: Preprocessing Data...")
    preprocess()

    print("Step 3: Generating Spectrogram...")
    generate_spectrogram()

    print("Step 4: Training Model...")
    train_model()

    print("Step 5: Making Predictions...")
    make_predictions()

    print("Project Completed Successfully!")

if __name__ == "__main__":
    main()