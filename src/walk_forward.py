import pandas as pd

from src.regime_detection import detect_regimes
from src.regime_labels import label_regimes
from src.strategy import generate_signals
from src.position_sizing import volatility_position_size
from src.backtest import run_position_sized_backtest


def walk_forward_backtest(
    df,
    train_size=756,
    test_size=126
):

    results = []
    period_results = []

    start = train_size

    while start < len(df):

        # ==========================================
        # 1. Create training and test windows
        # ==========================================

        train = df.iloc[
            start - train_size:start
        ].copy()

        test = df.iloc[
            start:min(
                start + test_size,
                len(df)
            )
        ].copy()

        if len(test) == 0:
            break

        # ==========================================
        # 2. Train regime model ONLY on training data
        # ==========================================

        train_regimes, model, scaler = (
            detect_regimes(train)
        )

        # ==========================================
        # 3. Label regimes using training data
        # ==========================================

        train_regimes, summary, labels = (
            label_regimes(train_regimes)
        )

        # ==========================================
        # 4. Prepare test features
        # ==========================================

        features = [
            "Return",
            "Volatility",
            "Momentum"
        ]

        X_test = test[features].copy()

        # ==========================================
        # 5. Clip test values using TRAIN statistics
        # ==========================================

        lower = train[features].quantile(0.01)
        upper = train[features].quantile(0.99)

        X_test = X_test.clip(
            lower=lower,
            upper=upper,
            axis=1
        )

        # ==========================================
        # 6. Scale test data using TRAIN scaler
        # ==========================================

        X_test_scaled = scaler.transform(
            X_test
        )

        # ==========================================
        # 7. Predict regimes on unseen test data
        # ==========================================

        test["Regime"] = (
            model.predict(X_test_scaled)
        )

        test["Regime_Label"] = (
            test["Regime"].map(labels)
        )

        # ==========================================
        # 8. Generate trading signals
        # ==========================================

        test = generate_signals(test)

        # ==========================================
        # 9. Apply volatility-based position sizing
        # ==========================================

        test = volatility_position_size(
            test,
            target_volatility=0.15
        )

        # ==========================================
        # 10. Backtest this test window
        # ==========================================

        test = run_position_sized_backtest(
            test
        )

        # ==========================================
        # 11. Calculate test-period metrics
        # ==========================================

        test_returns = (
            test["Net_Return"]
            .dropna()
        )

        if len(test_returns) > 0:

            cumulative_return = (
                (1 + test_returns).prod()
            ) - 1

            annual_volatility = (
                test_returns.std()
                * (252 ** 0.5)
            )

            sharpe = (
                test_returns.mean()
                / test_returns.std()
                * (252 ** 0.5)
                if test_returns.std() > 0
                else 0
            )

            # Calculate period drawdown
            equity = (
                1 + test_returns
            ).cumprod()

            running_peak = (
                equity.cummax()
            )

            drawdown = (
                equity / running_peak
            ) - 1

            max_drawdown = drawdown.min()

            win_rate = (
                (test_returns > 0).sum()
                / len(test_returns)
            )

            period_results.append({

                "Start": test.index.min(),

                "End": test.index.max(),

                "Return": cumulative_return,

                "Volatility": annual_volatility,

                "Sharpe": sharpe,

                "Max_Drawdown": max_drawdown,

                "Win_Rate": win_rate,

                "Observations": len(test_returns)
            })

        # ==========================================
        # 12. Store test results
        # ==========================================

        results.append(test)

        # ==========================================
        # 13. Move forward
        # ==========================================

        start += test_size

    # ==============================================
    # 14. Combine all test periods
    # ==============================================

    combined_results = pd.concat(
        results
    )

    period_df = pd.DataFrame(
        period_results
    )

    return combined_results, period_df
