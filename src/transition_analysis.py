import pandas as pd


def transition_matrix(df):

    transitions = pd.crosstab(
        df["Regime"],
        df["Regime"].shift(-1),
        normalize="index"
    )

    return transitions