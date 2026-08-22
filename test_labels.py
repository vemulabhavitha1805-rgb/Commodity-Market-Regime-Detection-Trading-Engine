from src.data_loader import load_data
from src.features import create_features
from src.regime_detection import detect_regimes
from src.regime_labels import label_regimes


# Load data
data = load_data()


# Create features
df = create_features(data)


# Detect regimes
df, model, scaler = detect_regimes(df)


# Label regimes
df, summary, labels = label_regimes(df)


print("\nRegime Statistics")
print("=" * 70)

print(summary)


print("\nRegime Labels")
print("=" * 70)

for regime, label in labels.items():
    print(f"Regime {regime}: {label}")


print("\nRegime Distribution")
print("=" * 70)

print(df["Regime_Label"].value_counts())


print("\nLatest Market State")
print("=" * 70)

print(
    df[
        [
            "Close",
            "Return",
            "Volatility",
            "Momentum",
            "Regime",
            "Regime_Label"
        ]
    ].tail(10)
)