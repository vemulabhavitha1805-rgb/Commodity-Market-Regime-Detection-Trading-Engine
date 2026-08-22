import pandas as pd


def label_regimes(df):

    summary = df.groupby("Regime")[
        [
            "Return",
            "Volatility",
            "Momentum"
        ]
    ].mean()

    labels = {}

    # Highest volatility regime
    high_vol_regime = summary["Volatility"].idxmax()

    labels[high_vol_regime] = "Extreme Volatility"

    # Remaining regimes
    remaining = summary.drop(index=high_vol_regime)

    # Highest return among remaining regimes
    bull_regime = remaining["Return"].idxmax()

    labels[bull_regime] = "Bull / Stable"

    # Remaining regime becomes bearish
    for regime in remaining.index:

        if regime != bull_regime:
            labels[regime] = "Bear / Moderate Volatility"

    result = df.copy()

    result["Regime_Label"] = result["Regime"].map(labels)

    return result, summary, labels