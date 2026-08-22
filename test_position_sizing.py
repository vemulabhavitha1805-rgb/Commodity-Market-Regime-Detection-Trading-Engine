from src.data_loader import load_data
from src.features import create_features
from src.regime_detection import detect_regimes
from src.regime_labels import label_regimes
from src.strategy import generate_signals
from src.position_sizing import volatility_position_size
from src.backtest import (
    run_backtest,
    run_position_sized_backtest,
    calculate_metrics
)


# ==========================================
# 1. Load data
# ==========================================

data = load_data()


# ==========================================
# 2. Feature engineering
# ==========================================

df = create_features(data)


# ==========================================
# 3. Detect regimes
# ==========================================

df, model, scaler = detect_regimes(df)


# ==========================================
# 4. Label regimes
# ==========================================

df, summary, labels = label_regimes(df)


# ==========================================
# 5. Generate trading signals
# ==========================================

df = generate_signals(df)


# ==========================================
# 6. Fixed-position strategy
# ==========================================

fixed_df = run_backtest(
    df.copy()
)


# ==========================================
# 7. Volatility position sizing
# ==========================================

sized_df = volatility_position_size(
    df.copy(),
    target_volatility=0.15
)


# ==========================================
# 8. Position-sized backtest
# ==========================================

sized_df = run_position_sized_backtest(
    sized_df
)


# ==========================================
# 9. Calculate metrics
# ==========================================

fixed_metrics = calculate_metrics(
    fixed_df,
    "Net_Return"
)

sized_metrics = calculate_metrics(
    sized_df,
    "Net_Return"
)


# ==========================================
# 10. Print results
# ==========================================

print("\n========================================")
print("FIXED POSITION STRATEGY")
print("========================================")

for metric, value in fixed_metrics.items():

    print(
        f"{metric}: {value:.4f}"
    )


print("\n========================================")
print("VOLATILITY-ADJUSTED STRATEGY")
print("========================================")

for metric, value in sized_metrics.items():

    print(
        f"{metric}: {value:.4f}"
    )


print("\n========================================")
print("FINAL PORTFOLIO VALUES")
print("========================================")

print(
    f"Fixed Position: "
    f"${fixed_df['Equity'].iloc[-1]:,.2f}"
)

print(
    f"Volatility Adjusted: "
    f"${sized_df['Equity'].iloc[-1]:,.2f}"
)