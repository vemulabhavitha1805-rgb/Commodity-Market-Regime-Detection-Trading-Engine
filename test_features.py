from src.data_loader import download_data
from src.features import create_features


# Download raw market data
data = download_data()

# Create trading features
df = create_features(data)


print("\nFirst 5 rows:")
print(df.head())


print("\nFeature columns:")
print(df.columns)


print("\nDataset shape:")
print(df.shape)


print("\nFeature statistics:")
print(
    df[
        [
            "Return",
            "Volatility",
            "Momentum",
            "MA20",
            "MA50",
            "Volume_Change"
        ]
    ].describe()
)