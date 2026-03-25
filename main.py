from src.data_collection import fetch_data
from src.preprocessing import preprocess
from src.stft_processing import generate_spectrogram
from src.train import train_model
from src.predict import predict

fetch_data()
preprocess()
generate_spectrogram()
train_model()
predict()