import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_PATH = "data/processed/real_estate_clean.csv"


def build_features():
    try:
        logger.info("Building features")

        df = pd.read_csv(DATA_PATH)

        # Convert property_type into dummy variables
        df = pd.get_dummies(df, columns=["property_type"], drop_first=True)

        # Separate features and target
        X = df.drop("price", axis=1)
        y = df["price"]

        return df, X, y

    except Exception as e:
        logger.error(f"build_features error: {e}")
        raise
