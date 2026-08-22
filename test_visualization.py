from src.data_loader import load_data
from src.features import create_features
from src.walk_forward import walk_forward_backtest
from src.visualization import (
    plot_regimes,
    plot_equity_curve,
    plot_drawdown,
    plot_position_size
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
# 3. Walk-forward validation
# ==========================================

results, period_results = (
    walk_forward_backtest(
        df,
        train_size=756,
        test_size=126
    )
)


# ==========================================
# 4. Regime chart
# ==========================================

plot_regimes(results)


# ==========================================
# 5. Equity curves
# ==========================================

strategy_equity = results["Equity"]


# Normalize buy & hold
initial_value = strategy_equity.iloc[0]

benchmark = (
    results["Close"]
    / results["Close"].iloc[0]
    * initial_value
)


plot_equity_curve(
    strategy_equity,
    benchmark
)


# ==========================================
# 6. Drawdown
# ==========================================

plot_drawdown(
    strategy_equity
)


# ==========================================
# 7. Position sizing
# ==========================================

plot_position_size(
    results
)