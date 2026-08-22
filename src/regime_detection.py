import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


REGIME_FEATURES = [
    "Return",
    "Volatility",
    "Momentum"
]


def detect_regimes(df, n_regimes=3):

    X = df[REGIME_FEATURES].copy()

    # Remove extreme observations
    lower = X.quantile(0.01)
    upper = X.quantile(0.99)

    X = X.clip(lower=lower, upper=upper, axis=1)

    # Standardize features
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # Gaussian Mixture Model
    model = GaussianMixture(
        n_components=n_regimes,
        covariance_type="full",
        n_init=10,
        random_state=42
    )

    regimes = model.fit_predict(X_scaled)

    result = df.copy()

    result["Regime"] = regimes

    return result, model, scaler