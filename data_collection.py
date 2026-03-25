import yfinance as yf
import pandas as pd
import os

def fetch_data():
    stocks = ['TCS.NS', 'INFY.NS', 'RELIANCE.NS']
    data = {}

    for stock in stocks:
        df = yf.download(stock, start="2020-01-01", end="2024-01-01")
        data[stock] = df['Close']

    df = pd.DataFrame(data)
    df.dropna(inplace=True)

    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/stock_data.csv")

    print("Data saved to data/raw/")

if __name__ == "__main__":
    fetch_data()