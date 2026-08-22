import numpy as np


def volatility_position_size(
    df,
    target_volatility=0.15
):

    result = df.copy()

    # Annualized volatility
    annualized_volatility = (
        result["Volatility"] * np.sqrt(252)
    )

    # Avoid division by zero
    annualized_volatility = (
        annualized_volatility.clip(lower=0.01)
    )

    # Volatility-scaled position
    position_size = (
        target_volatility /
        annualized_volatility
    )

    # Limit maximum exposure
    position_size = position_size.clip(
        lower=0,
        upper=1
    )

    result["Position_Size"] = position_size

    result["Final_Position"] = (
        result["Signal"] *
        result["Position_Size"]
    )

    return result