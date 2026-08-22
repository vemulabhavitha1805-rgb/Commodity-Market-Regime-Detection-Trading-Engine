# 📈 Commodity Market Regime Detection & Trading Engine

> **Machine-learning-driven quantitative trading system for detecting commodity market regimes, generating regime-aware trading signals, managing portfolio risk, and evaluating strategy performance through walk-forward validation.**

---

## 🚀 Overview

Financial markets behave differently across periods of rising prices, declining prices, and extreme volatility.

This project builds an end-to-end **commodity quantitative trading engine** that identifies these hidden market conditions using an unsupervised machine learning model and dynamically adjusts trading exposure according to the detected regime.

The system combines:

**Market Data → Feature Engineering → Regime Detection → Signal Generation → Position Sizing → Backtesting → Risk Analysis → Walk-Forward Validation**

The platform also provides an interactive **Streamlit quant dashboard** for visualizing market regimes, trading signals, portfolio performance, drawdowns, and regime-level attribution.

---

## 🎯 Key Features

- 📊 **Commodity market data pipeline**
- 🧠 **Gaussian Mixture Model (GMM) regime detection**
- 📈 **Quantitative feature engineering**
- 🔵 **Bull / Stable regime detection**
- 🟠 **Bear / Moderate Volatility regime detection**
- 🔴 **Extreme Volatility regime detection**
- 🎯 **Regime-aware trading signals**
- ⚖️ **Volatility-adjusted position sizing**
- 💸 **Transaction cost modeling**
- 📉 **Portfolio drawdown analysis**
- 📊 **Buy & Hold benchmark comparison**
- 🔄 **Walk-forward validation**
- 🔬 **Regime performance attribution**
- 🔁 **Regime transition probability analysis**
- 🖥️ **Interactive Streamlit dashboard**

---

# 🏗️ System Architecture

```text
                    ┌──────────────────────┐
                    │   Commodity Data     │
                    │    Yahoo Finance     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Feature Engineering  │
                    │                      │
                    │ • Returns            │
                    │ • Volatility         │
                    │ • Momentum           │
                    │ • MA20 / MA50        │
                    │ • Volume Change       │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  Regime Detection    │
                    │                      │
                    │ Gaussian Mixture     │
                    │ Model (GMM)          │
                    └──────────┬───────────┘
                               │
                               ▼
             ┌────────────────────────────────────┐
             │       Market Regime Labels         │
             │                                    │
             │  Bull / Stable                     │
             │  Bear / Moderate Volatility        │
             │  Extreme Volatility                │
             └────────────────┬───────────────────┘
                              │
                              ▼
                    ┌──────────────────────┐
                    │ Trading Signal Engine │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Position Sizing      │
                    │                      │
                    │ Volatility Adjusted  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │     Backtesting      │
                    │                      │
                    │ • Transaction Costs  │
                    │ • Equity Curve       │
                    │ • Drawdown           │
                    └──────────┬───────────┘
                               │
                               ▼
              ┌─────────────────────────────────┐
              │     Performance Evaluation      │
              │                                 │
              │ CAGR • Sharpe • Volatility      │
              │ Drawdown • Win Rate             │
              └───────────────┬─────────────────┘
                              │
                              ▼
                    ┌──────────────────────┐
                    │ Walk-Forward Testing  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Regime Attribution   │
                    └──────────────────────┘