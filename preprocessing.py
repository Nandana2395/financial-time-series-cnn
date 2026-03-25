import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

def preprocess():
    df = pd.read_csv("data/raw/stock_data.csv", index_col=0)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)

    df_scaled = pd.DataFrame(scaled, columns=df.columns)

    os.makedirs("data/processed", exist_ok=True)
    df_scaled.to_csv("data/processed/processed_data.csv")

    print("Preprocessing done")

if __name__ == "__main__":
    preprocess()