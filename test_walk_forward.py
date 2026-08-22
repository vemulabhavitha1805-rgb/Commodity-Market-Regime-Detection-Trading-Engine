from src.data_loader import load_data
from src.features import create_features
from src.walk_forward import walk_forward_backtest
from src.backtest import calculate_metrics


# ==========================================
# 1. Load data
# ==========================================

data = load_data()


# ==========================================
# 2. Feature engineering
# ==========================================

df = create_features(data)


# ==========================================
# 3. Walk-forward validation
# ==========================================

print("\nRunning walk-forward validation...")
print("=" * 60)

results, period_results = walk_forward_backtest(
    df,
    train_size=756,
    test_size=126
)


# ==========================================
# 4. Overall performance
# ==========================================

metrics = calculate_metrics(
    results,
    "Net_Return"
)


print("\nWalk-Forward Performance")
print("=" * 60)

for metric, value in metrics.items():

    print(
        f"{metric}: "
        f"{value:.4f}"
    )


# ==========================================
# 5. Individual test periods
# ==========================================

print("\n")
print("=" * 80)
print("WALK-FORWARD PERIOD RESULTS")
print("=" * 80)

print(
    period_results.to_string(
        index=False
    )
)


# ==========================================
# 6. Period summary
# ==========================================

print("\n")
print("=" * 80)
print("PERIOD SUMMARY")
print("=" * 80)

print(
    f"Number of test periods: "
    f"{len(period_results)}"
)

print(
    f"Profitable periods: "
    f"{(period_results['Return'] > 0).sum()}"
)

print(
    f"Losing periods: "
    f"{(period_results['Return'] < 0).sum()}"
)

print(
    f"Average period return: "
    f"{period_results['Return'].mean():.4f}"
)

print(
    f"Median period return: "
    f"{period_results['Return'].median():.4f}"
)

print(
    f"Average Sharpe: "
    f"{period_results['Sharpe'].mean():.4f}"
)

print(
    f"Average Maximum Drawdown: "
    f"{period_results['Max_Drawdown'].mean():.4f}"
)

print(
    f"Average Win Rate: "
    f"{period_results['Win_Rate'].mean():.4f}"
)