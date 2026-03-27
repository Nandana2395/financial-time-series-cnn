import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

def preprocess():
    print("Loading raw data...")
    df = pd.read_csv("data/raw/stock_data.csv")

    print("Initial shape:", df.shape)

    # ✅ Drop Date column
    if "Date" in df.columns:
        df.drop(columns=["Date"], inplace=True)

    # ✅ Convert everything to numeric (IMPORTANT FIX)
    df = df.apply(pd.to_numeric, errors='coerce')

    # ✅ Fill missing values
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    # ✅ Drop any remaining NaN rows
    df.dropna(inplace=True)

    print("Shape after cleaning:", df.shape)

    # ✅ Scaling
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)

    os.makedirs("data/processed", exist_ok=True)
    np.save("data/processed/processed.npy", scaled)

    print("Preprocessing completed successfully!")