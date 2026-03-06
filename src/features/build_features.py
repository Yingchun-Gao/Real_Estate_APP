import pandas as pd

DATA_PATH = "data/processed/real_estate_clean.csv"


def build_features():

    df = pd.read_csv(DATA_PATH)

    # Convert property_type into dummy variables
    df = pd.get_dummies(df, columns=["property_type"], drop_first=True)

    # Separate features and target
    X = df.drop("price", axis=1)
    y = df["price"]

    return df, X, y
