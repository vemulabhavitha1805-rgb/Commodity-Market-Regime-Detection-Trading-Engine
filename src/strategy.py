import numpy as np


def generate_signals(df):

    result = df.copy()

    result["Signal"] = 0

    # Bull + positive momentum → LONG
    bull_condition = (
        (result["Regime_Label"] == "Bull / Stable") &
        (result["Momentum"] > 0)
    )

    result.loc[bull_condition, "Signal"] = 1

    # Bear + negative momentum → SHORT
    bear_condition = (
        (result["Regime_Label"] == "Bear / Moderate Volatility") &
        (result["Momentum"] < 0)
    )

    result.loc[bear_condition, "Signal"] = -1

    # Extreme volatility → FLAT
    extreme_condition = (
        result["Regime_Label"] == "Extreme Volatility"
    )

    result.loc[extreme_condition, "Signal"] = 0

    return result