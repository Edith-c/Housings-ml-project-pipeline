import pandas as pd


def load_and_prepare_data(
    path="datasets/housing.csv",
    config=None
):
    if config is not None and not isinstance(config, dict):
        raise ValueError("config must be a dictionary")

    df = pd.read_csv(path)

    # copy to avoid modifying original
    df = df.copy()

    # handle missing values
    df = df.dropna()

    # encode categorical columns
    categorical_cols = df.select_dtypes(include=["object"]).columns
    df = pd.get_dummies(df, columns=categorical_cols)

    # split features / target
    target = "median_house_value"

    X = df.drop(columns=[target])
    y = df[target]

    return X, y