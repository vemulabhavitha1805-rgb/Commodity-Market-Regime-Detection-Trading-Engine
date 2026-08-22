from src.data_loader import load_data
from src.features import create_features
from src.regime_detection import detect_regimes
from src.regime_labels import label_regimes
from src.strategy import generate_signals


# Load market data
data = load_data()


# Feature engineering
df = create_features(data)


# Detect regimes
df, model, scaler = detect_regimes(df)


# Label regimes
df, summary, labels = label_regimes(df)


# Generate trading signals
df = generate_signals(df)


print("\nSignal Distribution")
print("=" * 60)

print(df["Signal"].value_counts())


print("\nLatest Trading Signals")
print("=" * 60)

print(
    df[
        [
            "Close",
            "Momentum",
            "Regime_Label",
            "Signal"
        ]
    ].tail(20)
)