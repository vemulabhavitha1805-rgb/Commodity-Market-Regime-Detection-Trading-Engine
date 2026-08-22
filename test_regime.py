from src.data_loader import load_data
from src.features import create_features
from src.regime_detection import detect_regimes


# Step 1: Download data

data = load_data()

# Step 2: Create features
df = create_features(data)


# Step 3: Detect market regimes
df, model, scaler = detect_regimes(df)


print("\nRegime distribution:")
print(df["Regime"].value_counts())


print("\nAverage characteristics of each regime:")

regime_summary = df.groupby("Regime")[
    [
        "Return",
        "Volatility",
        "Momentum"
    ]
].mean()

print(regime_summary)


print("\nLast 10 observations:")
print(
    df[
        [
            "Close",
            "Return",
            "Volatility",
            "Momentum",
            "Regime"
        ]
    ].tail(10)
)