import numpy as np


def apply_transaction_costs(df, transaction_cost=0.0005):

    result = df.copy()

    # Change in position
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

    # Net strategy return
    result["Net_Strategy_Return"] = (
        result["Strategy_Return"]
        - result["Transaction_Cost"]
    )

    return result