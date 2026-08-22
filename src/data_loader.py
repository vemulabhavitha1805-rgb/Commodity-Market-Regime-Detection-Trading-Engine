import os
import pandas as pd
import yfinance as yf


DATA_PATH = "data/raw/wti_crude_oil.csv"


def download_data(ticker="CL=F", period="10y"):

    data = yf.download(
        ticker,
        period=period,
        auto_adjust=True
    )

    if hasattr(data.columns, "levels"):
        data.columns = data.columns.get_level_values(0)

    data = data.dropna()

    os.makedirs("data/raw", exist_ok=True)

    data.to_csv(DATA_PATH)

    return data


def load_data():

    data = pd.read_csv(
        DATA_PATH,
        index_col=0,
        parse_dates=True
    )

    return data


if __name__ == "__main__":

    data = download_data()

    print("\nFirst 5 rows:")
    print(data.head())

    print("\nShape:")
    print(data.shape)

    print("\nColumns:")
    print(data.columns)

    print("\nData saved successfully!")