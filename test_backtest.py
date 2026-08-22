from src.data_loader import load_data
from src.features import create_features
from src.regime_detection import detect_regimes
from src.regime_labels import label_regimes
from src.strategy import generate_signals

from src.backtest import (
    run_position_sized_backtest,
    calculate_metrics,
    calculate_price_benchmark,
    calculate_cost_analysis,
    calculate_risk_statistics
)


# ============================================================
# 1. LOAD DATA
# ============================================================

data = load_data()


# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================

df = create_features(data)


# ============================================================
# 3. REGIME DETECTION
# ============================================================

df, model, scaler = detect_regimes(df)


# ============================================================
# 4. REGIME LABELING
# ============================================================

df, summary, labels = label_regimes(df)


# ============================================================
# 5. GENERATE TRADING SIGNALS
# ============================================================

df = generate_signals(df)


# ============================================================
# 6. REALISTIC BACKTEST
# ============================================================

df = run_position_sized_backtest(
    df,
    initial_capital=100000,
    transaction_cost=0.0005,
    slippage=0.0002,
    max_position=0.50
)


# ============================================================
# 7. BUY & HOLD BENCHMARK
# ============================================================

df = calculate_price_benchmark(
    df,
    initial_capital=100000
)


# ============================================================
# 8. PERFORMANCE METRICS
# ============================================================

strategy_metrics = calculate_metrics(
    df,
    "Net_Return"
)

gross_metrics = calculate_metrics(
    df,
    "Gross_Return"
)

benchmark_metrics = calculate_metrics(
    df,
    "Benchmark_Return"
)


# ============================================================
# 9. COST ANALYSIS
# ============================================================

cost_analysis = calculate_cost_analysis(df)


# ============================================================
# 10. RISK ANALYSIS
# ============================================================

risk_statistics = calculate_risk_statistics(df)


# ============================================================
# 11. PRINT GROSS PERFORMANCE
# ============================================================

print("\n")
print("=" * 60)
print("GROSS STRATEGY PERFORMANCE")
print("=" * 60)

for metric, value in gross_metrics.items():

    print(
        f"{metric}: {value:.4f}"
    )


# ============================================================
# 12. PRINT NET PERFORMANCE
# ============================================================

print("\n")
print("=" * 60)
print("NET STRATEGY PERFORMANCE")
print("=" * 60)

for metric, value in strategy_metrics.items():

    print(
        f"{metric}: {value:.4f}"
    )


# ============================================================
# 13. PRINT BENCHMARK
# ============================================================

print("\n")
print("=" * 60)
print("BUY & HOLD BENCHMARK")
print("=" * 60)

for metric, value in benchmark_metrics.items():

    print(
        f"{metric}: {value:.4f}"
    )


# ============================================================
# 14. PRINT COST ANALYSIS
# ============================================================

print("\n")
print("=" * 60)
print("TRADING COST ANALYSIS")
print("=" * 60)

for metric, value in cost_analysis.items():

    print(
        f"{metric}: {value:.6f}"
    )


# ============================================================
# 15. PRINT RISK ANALYSIS
# ============================================================

print("\n")
print("=" * 60)
print("RISK STATISTICS")
print("=" * 60)

for metric, value in risk_statistics.items():

    print(
        f"{metric}: {value:.6f}"
    )


# ============================================================
# 16. FINAL PORTFOLIO VALUES
# ============================================================

print("\n")
print("=" * 60)
print("FINAL PORTFOLIO VALUES")
print("=" * 60)

print(
    f"Gross Strategy: "
    f"${df['Gross_Equity'].iloc[-1]:,.2f}"
)

print(
    f"Net Strategy: "
    f"${df['Equity'].iloc[-1]:,.2f}"
)

print(
    f"Buy & Hold: "
    f"${df['Benchmark_Equity'].iloc[-1]:,.2f}"
)


# ============================================================
# 17. COST IMPACT
# ============================================================

gross_final = df["Gross_Equity"].iloc[-1]
net_final = df["Equity"].iloc[-1]

cost_impact = (
    gross_final - net_final
)

print("\n")
print("=" * 60)
print("COST IMPACT")
print("=" * 60)

print(
    f"Gross Portfolio Value: "
    f"${gross_final:,.2f}"
)

print(
    f"Net Portfolio Value: "
    f"${net_final:,.2f}"
)

print(
    f"Cost Impact: "
    f"${cost_impact:,.2f}"
)

print("\n")
print("=" * 60)
print("BACKTEST COMPLETE")
print("=" * 60)