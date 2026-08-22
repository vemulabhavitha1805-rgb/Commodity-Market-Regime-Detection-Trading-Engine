import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from src.data_loader import load_data
from src.features import create_features
from src.regime_detection import detect_regimes
from src.regime_labels import label_regimes
from src.strategy import generate_signals
from src.backtest import (
    run_backtest,
    calculate_metrics,
    calculate_price_benchmark,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Commodity Quant Engine",
    page_icon="📈",
    layout="wide",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #080b10;
        color: #f5f7fa;
    }

    .main {
        background-color: #080b10;
    }

    h1, h2, h3 {
        color: #f5f7fa !important;
    }

    .metric-card {
        background-color: #10151d;
        border: 1px solid #202733;
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 10px;
    }

    .metric-title {
        color: #7d8795;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .metric-value {
        color: #ffffff;
        font-size: 26px;
        font-weight: 700;
        margin-top: 6px;
    }

    .regime-card {
        background-color: #10151d;
        border: 1px solid #202733;
        border-radius: 12px;
        padding: 22px;
        text-align: center;
        margin-bottom: 15px;
    }

    .regime-title {
        color: #7d8795;
        font-size: 12px;
        text-transform: uppercase;
    }

    .regime-value {
        color: #00d395;
        font-size: 28px;
        font-weight: 700;
        margin-top: 8px;
    }

    .signal-long {
        color: #00d395;
        font-size: 30px;
        font-weight: 700;
    }

    .signal-short {
        color: #ff5c6c;
        font-size: 30px;
        font-weight: 700;
    }

    .signal-flat {
        color: #f2c94c;
        font-size: 30px;
        font-weight: 700;
    }

    .section-title {
        color: #ffffff;
        font-size: 20px;
        font-weight: 700;
        margin-top: 30px;
        margin-bottom: 5px;
    }

    .section-description {
        color: #707a89;
        font-size: 13px;
        margin-bottom: 15px;
    }

    .footer {
        text-align: center;
        color: #596270;
        font-size: 12px;
        margin-top: 40px;
        padding: 20px;
        border-top: 1px solid #202733;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Engine Controls")

st.sidebar.markdown("### Market")

commodity = st.sidebar.selectbox(
    "Select Commodity",
    ["Crude Oil"],
)

st.sidebar.markdown("### Trading Parameters")

transaction_cost = st.sidebar.number_input(
    "Transaction Cost",
    min_value=0.0,
    max_value=0.01,
    value=0.0005,
    step=0.0001,
    format="%.4f",
)

initial_capital = st.sidebar.number_input(
    "Initial Capital ($)",
    min_value=1000,
    max_value=10000000,
    value=100000,
    step=10000,
)

st.sidebar.markdown("---")

st.sidebar.markdown("### Model")

st.sidebar.info(
    """
    **Regime Model**

    Gaussian Mixture Model

    **Features**

    • Return  
    • Volatility  
    • Momentum  
    • Moving Averages

    **Validation**

    Walk-Forward Validation
    """
)


# ============================================================
# LOAD AND PROCESS DATA
# ============================================================

@st.cache_data
def run_pipeline(
    transaction_cost,
    initial_capital,
):

    # Load market data
    data = load_data()

    # Feature engineering
    df = create_features(data)

    # Regime detection
    df, model, scaler = detect_regimes(df)

    # Regime labeling
    df, summary, labels = label_regimes(df)

    # Generate trading signals
    df = generate_signals(df)

    # Backtest
    df = run_backtest(
        df,
        initial_capital=initial_capital,
        transaction_cost=transaction_cost,
    )

    # Benchmark
    df = calculate_price_benchmark(
        df,
        initial_capital=initial_capital,
    )

    return df


with st.spinner("Running market intelligence engine..."):

    try:

        df = run_pipeline(
            transaction_cost,
            initial_capital,
        )

    except Exception as e:

        st.error("Unable to run the trading engine.")
        st.exception(e)
        st.stop()


# ============================================================
# HEADER
# ============================================================

st.title("📈 Commodity Market Regime Detection")

st.markdown(
    """
    **Quantitative trading intelligence platform**

    Detect market regimes using volatility, momentum and price
    behaviour, then evaluate systematic trading strategies.
    """
)


st.markdown("---")


# ============================================================
# CURRENT MARKET STATE
# ============================================================

latest = df.iloc[-1]

price = latest["Close"]
daily_return = latest["Return"]
volatility = latest["Volatility"]
momentum = latest["Momentum"]
regime = latest["Regime_Label"]


# Determine signal

if "Signal" in df.columns:

    signal_value = latest["Signal"]

    if signal_value > 0:
        signal = "LONG"
        signal_class = "signal-long"

    elif signal_value < 0:
        signal = "SHORT"
        signal_class = "signal-short"

    else:
        signal = "FLAT"
        signal_class = "signal-flat"

else:

    signal = "N/A"
    signal_class = "signal-flat"


st.markdown(
    '<div class="section-title">Current Market State</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-description">'
    'Latest detected market conditions'
    '</div>',
    unsafe_allow_html=True,
)


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.markdown(
        f"""
        <div class="metric-card">

        <div class="metric-title">
        Crude Oil Price
        </div>

        <div class="metric-value">
        ${price:,.2f}
        </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


with col2:

    st.markdown(
        f"""
        <div class="metric-card">

        <div class="metric-title">
        Daily Return
        </div>

        <div class="metric-value">
        {daily_return:.2%}
        </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


with col3:

    st.markdown(
        f"""
        <div class="metric-card">

        <div class="metric-title">
        Volatility
        </div>

        <div class="metric-value">
        {volatility:.2%}
        </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


with col4:

    st.markdown(
        f"""
        <div class="metric-card">

        <div class="metric-title">
        Momentum
        </div>

        <div class="metric-value">
        {momentum:.2%}
        </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# REGIME + SIGNAL
# ============================================================

col1, col2 = st.columns(2)


with col1:

    st.markdown(
        '<div class="section-title">Market Regime</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="regime-card">

        <div class="regime-title">
        Detected Regime
        </div>

        <div class="regime-value">
        {regime}
        </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


with col2:

    st.markdown(
        '<div class="section-title">Trading Signal</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="regime-card">

        <div class="regime-title">
        Current Position
        </div>

        <div class="{signal_class}">
        {signal}
        </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# PERFORMANCE
# ============================================================

strategy_metrics = calculate_metrics(
    df,
    "Net_Return",
)

benchmark_metrics = calculate_metrics(
    df,
    "Benchmark_Return",
)


st.markdown(
    '<div class="section-title">Strategy Performance</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-description">'
    'Risk-adjusted performance of the regime trading strategy'
    '</div>',
    unsafe_allow_html=True,
)


col1, col2, col3, col4, col5 = st.columns(5)


performance_data = [
    ("CAGR", strategy_metrics["CAGR"]),
    ("Sharpe Ratio", strategy_metrics["Sharpe Ratio"]),
    ("Annual Volatility", strategy_metrics["Annual Volatility"]),
    ("Maximum Drawdown", strategy_metrics["Maximum Drawdown"]),
    ("Win Rate", strategy_metrics["Win Rate"]),
]


for column, (name, value) in zip(
    [col1, col2, col3, col4, col5],
    performance_data,
):

    with column:

        st.metric(
            label=name,
            value=f"{value:.2%}"
            if name != "Sharpe Ratio"
            else f"{value:.2f}",
        )


# ============================================================
# PRICE CHART
# ============================================================

st.markdown(
    '<div class="section-title">Commodity Price</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-description">'
    'Historical crude oil price'
    '</div>',
    unsafe_allow_html=True,
)


fig_price = go.Figure()


fig_price.add_trace(
    go.Scatter(
        x=df.index,
        y=df["Close"],
        mode="lines",
        name="Crude Oil",
        line=dict(
            width=2,
        ),
    )
)


fig_price.update_layout(
    template="plotly_dark",
    height=450,
    margin=dict(
        l=20,
        r=20,
        t=20,
        b=20,
    ),
    xaxis_title="Date",
    yaxis_title="Price ($)",
    hovermode="x unified",
)


st.plotly_chart(
    fig_price,
    use_container_width=True,
)


# ============================================================
# REGIME DISTRIBUTION
# ============================================================

col1, col2 = st.columns(2)


with col1:

    st.markdown(
        '<div class="section-title">Regime Distribution</div>',
        unsafe_allow_html=True,
    )

    regime_counts = (
        df["Regime_Label"]
        .value_counts()
    )


    fig_regime = go.Figure(
        data=[
            go.Pie(
                labels=regime_counts.index,
                values=regime_counts.values,
                hole=0.55,
            )
        ]
    )


    fig_regime.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(
            l=20,
            r=20,
            t=20,
            b=20,
        ),
    )


    st.plotly_chart(
        fig_regime,
        use_container_width=True,
    )


# ============================================================
# EQUITY CURVE
# ============================================================

with col2:

    st.markdown(
        '<div class="section-title">Strategy vs Benchmark</div>',
        unsafe_allow_html=True,
    )

    fig_equity = go.Figure()


    fig_equity.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Equity"],
            mode="lines",
            name="Regime Strategy",
            line=dict(
                width=2,
            ),
        )
    )


    fig_equity.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Benchmark_Equity"],
            mode="lines",
            name="Buy & Hold",
            line=dict(
                width=2,
                dash="dash",
            ),
        )
    )


    fig_equity.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(
            l=20,
            r=20,
            t=20,
            b=20,
        ),
        yaxis_title="Portfolio Value ($)",
        hovermode="x unified",
    )


    st.plotly_chart(
        fig_equity,
        use_container_width=True,
    )


# ============================================================
# DRAWDOWN
# ============================================================

st.markdown(
    '<div class="section-title">Portfolio Drawdown</div>',
    unsafe_allow_html=True,
)

drawdown = df["Drawdown"] * 100


fig_drawdown = go.Figure()


fig_drawdown.add_trace(
    go.Scatter(
        x=df.index,
        y=drawdown,
        mode="lines",
        fill="tozeroy",
        name="Drawdown",
    )
)


fig_drawdown.update_layout(
    template="plotly_dark",
    height=350,
    margin=dict(
        l=20,
        r=20,
        t=20,
        b=20,
    ),
    yaxis_title="Drawdown (%)",
)


st.plotly_chart(
    fig_drawdown,
    use_container_width=True,
)


# ============================================================
# REGIME PERFORMANCE
# ============================================================

st.markdown(
    '<div class="section-title">Performance by Regime</div>',
    unsafe_allow_html=True,
)

regime_returns = (
    df.groupby("Regime_Label")["Net_Return"]
    .apply(
        lambda x: (1 + x).prod() - 1
    )
    .sort_values()
)


fig_regime_perf = go.Figure()


fig_regime_perf.add_trace(
    go.Bar(
        x=regime_returns.values * 100,
        y=regime_returns.index,
        orientation="h",
        text=[
            f"{x:.2f}%"
            for x in regime_returns.values * 100
        ],
        textposition="outside",
    )
)


fig_regime_perf.update_layout(
    template="plotly_dark",
    height=400,
    margin=dict(
        l=20,
        r=50,
        t=20,
        b=20,
    ),
    xaxis_title="Cumulative Return (%)",
)


st.plotly_chart(
    fig_regime_perf,
    use_container_width=True,
)


# ============================================================
# BENCHMARK COMPARISON
# ============================================================

st.markdown(
    '<div class="section-title">Benchmark Comparison</div>',
    unsafe_allow_html=True,
)


comparison = pd.DataFrame(
    {
        "Metric": [
            "CAGR",
            "Annual Volatility",
            "Sharpe Ratio",
            "Maximum Drawdown",
            "Win Rate",
        ],
        "Regime Strategy": [
            strategy_metrics["CAGR"],
            strategy_metrics["Annual Volatility"],
            strategy_metrics["Sharpe Ratio"],
            strategy_metrics["Maximum Drawdown"],
            strategy_metrics["Win Rate"],
        ],
        "Buy & Hold": [
            benchmark_metrics["CAGR"],
            benchmark_metrics["Annual Volatility"],
            benchmark_metrics["Sharpe Ratio"],
            benchmark_metrics["Maximum Drawdown"],
            benchmark_metrics["Win Rate"],
        ],
    }
)


st.dataframe(
    comparison,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# FINAL PORTFOLIO
# ============================================================

strategy_final = df["Equity"].iloc[-1]
benchmark_final = df["Benchmark_Equity"].iloc[-1]


col1, col2 = st.columns(2)


with col1:

    st.metric(
        "Strategy Final Value",
        f"${strategy_final:,.2f}",
        f"{strategy_final - initial_capital:+,.2f}",
    )


with col2:

    st.metric(
        "Buy & Hold Final Value",
        f"${benchmark_final:,.2f}",
        f"{benchmark_final - initial_capital:+,.2f}",
    )


# ============================================================
# RECENT DATA
# ============================================================

with st.expander("View Recent Market Data"):

    display_columns = [
        "Close",
        "Return",
        "Volatility",
        "Momentum",
        "Regime_Label",
        "Signal",
        "Net_Return",
        "Equity",
        "Drawdown",
    ]

    display_columns = [
        col
        for col in display_columns
        if col in df.columns
    ]

    st.dataframe(
        df[display_columns]
        .tail(20)
        .sort_index(
            ascending=False
        ),
        use_container_width=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

    Commodity Market Regime Detection & Trading Engine

    <br>

    Gaussian Mixture Model • Feature Engineering •
    Regime-Based Trading • Backtesting

    <br><br>

    Research and educational project — not financial advice.

    </div>
    """,
    unsafe_allow_html=True,
)