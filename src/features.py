import pandas as pd


def create_features(data):

    df = data.copy()

    # 1. Daily return
    df["Return"] = df["Close"].pct_change()

    # 2. 20-day rolling volatility
    df["Volatility"] = (
        df["Return"]
        .rolling(window=20)
        .std()
    )

    # 3. 20-day momentum
    df["Momentum"] = (
        df["Close"] / df["Close"].shift(20)
    ) - 1

    # 4. Short-term moving average
    df["MA20"] = (
        df["Close"]
        .rolling(window=20)
        .mean()
    )

    # 5. Medium-term moving average
    df["MA50"] = (
        df["Close"]
        .rolling(window=50)
        .mean()
    )

    # 6. Volume percentage change
    df["Volume_Change"] = (
        df["Volume"].pct_change()
    )

    # Remove rows created by rolling calculations
    df = df.dropna()

    return df