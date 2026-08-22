import numpy as np
import pandas as pd
from src.position_sizing import volatility_position_size


# ============================================================
# MAIN POSITION-SIZED BACKTEST
# ============================================================

def run_position_sized_backtest(
    df,
    initial_capital=100000,
    transaction_cost=0.0005,
    slippage=0.0002,
    max_position=0.50
):
    """
    Run realistic backtest with:

    - Next-day execution
    - Transaction costs
    - Slippage
    - Position limits
    - Turnover tracking
    - Gross and net returns
    - Equity curve
    - Drawdown
    """

    result = df.copy()

    if "Final_Position" not in result.columns:
        result = volatility_position_size(result)

    # --------------------------------------------------------
    # 1. EXECUTE POSITION ON FOLLOWING DAY
    # --------------------------------------------------------

    result["Executed_Position"] = (
        result["Final_Position"]
        .shift(1)
        .fillna(0)
    )

    # --------------------------------------------------------
    # 2. POSITION LIMIT
    # --------------------------------------------------------

    result["Executed_Position"] = (
        result["Executed_Position"]
        .clip(
            -max_position,
            max_position
        )
    )

    # --------------------------------------------------------
    # 3. GROSS STRATEGY RETURN
    # --------------------------------------------------------

    result["Gross_Return"] = (
        result["Executed_Position"]
        * result["Return"]
    )

    # --------------------------------------------------------
    # 4. POSITION CHANGE / TURNOVER
    # --------------------------------------------------------

    result["Position_Change"] = (
        result["Executed_Position"]
        .diff()
        .abs()
        .fillna(0)
    )

    # --------------------------------------------------------
    # 5. TRANSACTION COST
    # --------------------------------------------------------

    result["Transaction_Cost"] = (
        result["Position_Change"]
        * transaction_cost
    )

    # --------------------------------------------------------
    # 6. SLIPPAGE
    # --------------------------------------------------------

    result["Slippage_Cost"] = (
        result["Position_Change"]
        * slippage
    )

    # --------------------------------------------------------
    # 7. TOTAL TRADING COST
    # --------------------------------------------------------

    result["Total_Trading_Cost"] = (
        result["Transaction_Cost"]
        + result["Slippage_Cost"]
    )

    # --------------------------------------------------------
    # 8. NET RETURN
    # --------------------------------------------------------

    result["Net_Return"] = (
        result["Gross_Return"]
        - result["Total_Trading_Cost"]
    )

    # --------------------------------------------------------
    # 9. EQUITY CURVES
    # --------------------------------------------------------

    result["Gross_Equity"] = (
        initial_capital
        * (1 + result["Gross_Return"]).cumprod()
    )

    result["Equity"] = (
        initial_capital
        * (1 + result["Net_Return"]).cumprod()
    )

    # --------------------------------------------------------
    # 10. RUNNING PEAK
    # --------------------------------------------------------

    result["Running_Peak"] = (
        result["Equity"].cummax()
    )

    # --------------------------------------------------------
    # 11. DRAWDOWN
    # --------------------------------------------------------

    result["Drawdown"] = (
        result["Equity"]
        / result["Running_Peak"]
    ) - 1

    # --------------------------------------------------------
    # 12. TURNOVER
    # --------------------------------------------------------

    result["Turnover"] = (
        result["Position_Change"]
    )

    # --------------------------------------------------------
    # 13. CUMULATIVE COST
    # --------------------------------------------------------

    result["Cumulative_Transaction_Cost"] = (
        result["Transaction_Cost"].cumsum()
    )

    result["Cumulative_Slippage"] = (
        result["Slippage_Cost"].cumsum()
    )

    result["Cumulative_Trading_Cost"] = (
        result["Total_Trading_Cost"].cumsum()
    )

    return result


# ============================================================
# GENERIC SIGNAL BACKTEST
# ============================================================

def run_backtest(
    df,
    initial_capital=100000,
    transaction_cost=0.0005,
    slippage=0.0002,
    max_position=1.0
):

    result = df.copy()

    # Execute signal next trading day
    result["Position"] = (
        result["Signal"]
        .shift(1)
        .fillna(0)
    )

    # Position limit
    result["Position"] = (
        result["Position"]
        .clip(
            -max_position,
            max_position
        )
    )

    # Gross return
    result["Gross_Return"] = (
        result["Position"]
        * result["Return"]
    )

    # Position change
    result["Position_Change"] = (
        result["Position"]
        .diff()
        .abs()
        .fillna(0)
    )

    # Transaction cost
    result["Transaction_Cost"] = (
        result["Position_Change"]
        * transaction_cost
    )

    # Slippage
    result["Slippage_Cost"] = (
        result["Position_Change"]
        * slippage
    )

    # Total cost
    result["Total_Trading_Cost"] = (
        result["Transaction_Cost"]
        + result["Slippage_Cost"]
    )

    # Net return
    result["Net_Return"] = (
        result["Gross_Return"]
        - result["Total_Trading_Cost"]
    )

    # Gross equity
    result["Gross_Equity"] = (
        initial_capital
        * (1 + result["Gross_Return"]).cumprod()
    )

    # Net equity
    result["Equity"] = (
        initial_capital
        * (1 + result["Net_Return"]).cumprod()
    )

    # Running peak
    result["Running_Peak"] = (
        result["Equity"].cummax()
    )

    # Drawdown
    result["Drawdown"] = (
        result["Equity"]
        / result["Running_Peak"]
    ) - 1

    return result


# ============================================================
# PERFORMANCE METRICS
# ============================================================

def calculate_metrics(
    df,
    return_column="Net_Return"
):

    returns = (
        df[return_column]
        .dropna()
    )

    total_days = len(returns)

    if total_days == 0:
        return {}

    # --------------------------------------------------------
    # CAGR
    # --------------------------------------------------------

    years = total_days / 252

    total_growth = (
        1 + returns
    ).prod()

    cagr = (
        total_growth
        ** (1 / years)
    ) - 1

    # --------------------------------------------------------
    # ANNUAL VOLATILITY
    # --------------------------------------------------------

    annual_volatility = (
        returns.std()
        * np.sqrt(252)
    )

    # --------------------------------------------------------
    # SHARPE RATIO
    #
    # Mean daily return / daily std × sqrt(252)
    # --------------------------------------------------------

    if returns.std() > 0:

        sharpe = (
            returns.mean()
            / returns.std()
            * np.sqrt(252)
        )

    else:

        sharpe = 0

    # --------------------------------------------------------
    # EQUITY
    # --------------------------------------------------------

    equity = (
        1 + returns
    ).cumprod()

    running_peak = (
        equity.cummax()
    )

    drawdown = (
        equity
        / running_peak
    ) - 1

    max_drawdown = (
        drawdown.min()
    )

    # --------------------------------------------------------
    # WIN RATE
    # --------------------------------------------------------

    win_rate = (
        returns > 0
    ).mean()

    # --------------------------------------------------------
    # TOTAL RETURN
    # --------------------------------------------------------

    total_return = (
        total_growth - 1
    )

    return {

        "CAGR": cagr,

        "Total Return": total_return,

        "Annual Volatility":
            annual_volatility,

        "Sharpe Ratio":
            sharpe,

        "Maximum Drawdown":
            max_drawdown,

        "Win Rate":
            win_rate
    }


# ============================================================
# PRICE BENCHMARK
# ============================================================

def calculate_price_benchmark(
    df,
    initial_capital=100000
):

    result = df.copy()

    # Make sure prices are valid
    valid_prices = (
        (result["Close"] > 0)
        &
        (result["Close"].shift(1) > 0)
    )

    benchmark_return = (
        result["Return"]
        .where(valid_prices)
    )

    benchmark_return = (
        benchmark_return
        .fillna(0)
    )

    result["Benchmark_Return"] = (
        benchmark_return
    )

    # Benchmark equity
    result["Benchmark_Equity"] = (
        initial_capital
        *
        (1 + result["Benchmark_Return"])
        .cumprod()
    )

    # Running benchmark peak
    running_peak = (
        result["Benchmark_Equity"]
        .cummax()
    )

    # Benchmark drawdown
    result["Benchmark_Drawdown"] = (
        result["Benchmark_Equity"]
        / running_peak
    ) - 1

    return result


# ============================================================
# COST ANALYSIS
# ============================================================

def calculate_cost_analysis(df):

    analysis = {}

    # Total transaction costs
    analysis["Transaction Costs"] = (
        df["Transaction_Cost"]
        .sum()
    )

    # Total slippage
    analysis["Slippage"] = (
        df["Slippage_Cost"]
        .sum()
    )

    # Total trading costs
    analysis["Total Trading Costs"] = (
        df["Total_Trading_Cost"]
        .sum()
    )

    # Total turnover
    analysis["Total Turnover"] = (
        df["Position_Change"]
        .sum()
    )

    # Average daily turnover
    analysis["Average Daily Turnover"] = (
        df["Position_Change"]
        .mean()
    )

    # Number of trades
    analysis["Number of Position Changes"] = (
        (df["Position_Change"] > 0)
        .sum()
    )

    return analysis


# ============================================================
# RISK ANALYSIS
# ============================================================

def calculate_risk_statistics(df):

    returns = (
        df["Net_Return"]
        .dropna()
    )

    statistics = {}

    # Best day
    statistics["Best Day"] = (
        returns.max()
    )

    # Worst day
    statistics["Worst Day"] = (
        returns.min()
    )

    # Average return
    statistics["Average Daily Return"] = (
        returns.mean()
    )

    # Return standard deviation
    statistics["Daily Volatility"] = (
        returns.std()
    )

    # 95% VaR
    statistics["VaR 95%"] = (
        returns.quantile(0.05)
    )

    # 99% VaR
    statistics["VaR 99%"] = (
        returns.quantile(0.01)
    )

    return statistics