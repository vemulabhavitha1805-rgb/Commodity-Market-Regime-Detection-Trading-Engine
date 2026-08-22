from src.data_loader import download_data
from src.features import create_features


# Download data
data = download_data()

# Create features
df = create_features(data)


features = [
    "Return",
    "Volatility",
    "Momentum",
    "MA20",
    "MA50",
    "Volume_Change"
]


print("\nFeature Statistics")
print("=" * 60)

print(df[features].describe().T)


print("\nMissing Values")
print("=" * 60)

print(df[features].isnull().sum())


print("\nExtreme Values")
print("=" * 60)

for feature in features:

    print(f"\n{feature}")

    print("Minimum:", df[feature].min())
    print("Maximum:", df[feature].max())