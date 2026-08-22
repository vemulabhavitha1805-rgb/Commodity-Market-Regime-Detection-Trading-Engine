import numpy as np

from src.data_loader import load_data
from src.features import create_features
from src.walk_forward import walk_forward_backtest


# ==========================================
# 1. Load data
# ==========================================

data = load_data()


# ==========================================
# 2. Feature engineering
# ==========================================

df = create_features(data)


# ==========================================
# 3. Run walk-forward validation
# ==========================================

print("\nRunning walk-forward validation...")

results, period_results = walk_forward_backtest(
    df,
    train_size=756,
    test_size=126
)


# ==========================================
# 4. Regime performance attribution
# ==========================================

print("\n")
print("=" * 80)
print("REGIME PERFORMANCE ATTRIBUTION")
print("=" * 80)


regimes = results["Regime_Label"].dropna().unique()


for regime in regimes:

    regime_data = results[
        results["Regime_Label"] == regime
    ].copy()

    returns = (
        regime_data["Net_Return"]
        .dropna()
    )

    if len(returns) == 0:
        continue

    # Cumulative return
    cumulative_return = (
        (1 + returns).prod()
    ) - 1

    # Annualized volatility
    annual_volatility = (
        returns.std()
        * np.sqrt(252)
    )

    # Sharpe ratio
    if returns.std() > 0:

        sharpe = (
            returns.mean()
            / returns.std()
            * np.sqrt(252)
        )

    else:

        sharpe = 0

    # Win rate
    win_rate = (
        (returns > 0).sum()
        / len(returns)
    )

    # Equity curve
    equity = (
        1 + returns
    ).cumprod()

    running_peak = (
        equity.cummax()
    )

    drawdown = (
        equity / running_peak
    ) - 1

    max_drawdown = drawdown.min()

    # Average position
    average_position = (
        regime_data["Final_Position"]
        .abs()
        .mean()
    )

    print(f"\nRegime: {regime}")
    print("-" * 60)

    print(
        f"Observations: "
        f"{len(returns)}"
    )

    print(
        f"Cumulative Return: "
        f"{cumulative_return:.4f}"
    )

    print(
        f"Annual Volatility: "
        f"{annual_volatility:.4f}"
    )

    print(
        f"Sharpe Ratio: "
        f"{sharpe:.4f}"
    )

    print(
        f"Maximum Drawdown: "
        f"{max_drawdown:.4f}"
    )

    print(
        f"Win Rate: "
        f"{win_rate:.4f}"
    )

    print(
        f"Average Position Size: "
        f"{average_position:.4f}"
    )


# ==========================================
# 5. Regime distribution
# ==========================================

print("\n")
print("=" * 80)
print("REGIME DISTRIBUTION")
print("=" * 80)

distribution = (
    results["Regime_Label"]
    .value_counts()
)

print(distribution)


# ==========================================
# 6. Return contribution
# ==========================================

print("\n")
print("=" * 80)
print("RETURN CONTRIBUTION BY REGIME")
print("=" * 80)

for regime in regimes:

    regime_data = results[
        results["Regime_Label"] == regime
    ]

    regime_return = (
        (1 + regime_data["Net_Return"].dropna()).prod()
    ) - 1

    print(
        f"{regime}: "
        f"{regime_return:.4f}"
    )