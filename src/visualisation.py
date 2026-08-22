import matplotlib.pyplot as plt
import pandas as pd


def plot_regimes(df):

    plt.figure(figsize=(15, 7))

    labels = df["Regime_Label"].dropna().unique()

    for label in labels:

        data = df[
            df["Regime_Label"] == label
        ]

        plt.scatter(
            data.index,
            data["Close"],
            s=8,
            label=label
        )

    plt.plot(
        df.index,
        df["Close"],
        alpha=0.35,
        linewidth=1
    )

    plt.title(
        "WTI Crude Oil Price by Market Regime"
    )

    plt.xlabel("Date")
    plt.ylabel("Price ($)")

    plt.legend()

    plt.grid(alpha=0.2)

    plt.tight_layout()

    plt.show()


def plot_equity_curve(
    strategy,
    benchmark
):

    plt.figure(figsize=(15, 7))

    plt.plot(
        strategy.index,
        strategy,
        label="Regime Trading Strategy",
        linewidth=2
    )

    plt.plot(
        benchmark.index,
        benchmark,
        label="Buy & Hold",
        linewidth=2
    )

    plt.title(
        "Strategy vs Buy & Hold"
    )

    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")

    plt.legend()

    plt.grid(alpha=0.2)

    plt.tight_layout()

    plt.show()


def plot_drawdown(
    equity
):

    running_peak = equity.cummax()

    drawdown = (
        equity / running_peak
    ) - 1

    plt.figure(figsize=(15, 6))

    plt.plot(
        drawdown.index,
        drawdown,
        linewidth=1.5
    )

    plt.fill_between(
        drawdown.index,
        drawdown,
        0,
        alpha=0.25
    )

    plt.title(
        "Strategy Drawdown"
    )

    plt.xlabel("Date")
    plt.ylabel("Drawdown")

    plt.grid(alpha=0.2)

    plt.tight_layout()

    plt.show()


def plot_position_size(df):

    plt.figure(figsize=(15, 6))

    plt.plot(
        df.index,
        df["Final_Position"],
        linewidth=1.2
    )

    plt.axhline(
        0,
        linestyle="--",
        linewidth=1
    )

    plt.title(
        "Volatility-Adjusted Position Size"
    )

    plt.xlabel("Date")
    plt.ylabel("Position")

    plt.grid(alpha=0.2)

    plt.tight_layout()

    plt.show()