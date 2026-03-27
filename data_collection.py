import yfinance as yf
import pandas as pd
import os

def fetch_data():
    stocks = ["TCS.NS", "INFY.NS", "RELIANCE.NS"]
    final_df = pd.DataFrame()

    for stock in stocks:
        print(f"Downloading {stock}...")
        data = yf.download(stock, start="2020-01-01", end="2024-01-01")

        df = data[['Close']].copy()
        df.rename(columns={'Close': stock}, inplace=True)

        if final_df.empty:
            final_df = df
        else:
            final_df = pd.merge(final_df, df, left_index=True, right_index=True, how='inner')

    final_df.reset_index(inplace=True)

    os.makedirs("data/raw", exist_ok=True)
    final_df.to_csv("data/raw/stock_data.csv", index=False)

    print("Data saved successfully!", final_df.shape)