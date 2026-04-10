import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RAW_PATH = "data/raw/real_estate.csv"
PROCESSED_PATH = "data/processed/real_estate_clean.csv"


def clean_data(df):
    try:
        # Remove duplicate rows
        df = df.drop_duplicates()

        # Convert basement column to numeric
        df["basement"] = pd.to_numeric(df["basement"], errors="coerce")

        # Replace missing basement values with 0
        df["basement"] = df["basement"].fillna(0).astype(int)

        # Feature engineering
        df["popular"] = ((df["beds"] == 2) & (df["baths"] == 2)).astype(int)
        df["recession"] = df["year_sold"].between(2010, 2013).astype(int)
        df["property_age"] = df["year_sold"] - df["year_built"]

        # Remove invalid records where property age is negative
        df = df[df["property_age"] >= 0]

        return df

    except Exception as e:
        logger.error(f"clean_data error: {e}")
        raise


def main():
    try:
        logger.info("Loading raw data")
        df = pd.read_csv(RAW_PATH)

        df_clean = clean_data(df)

        # Create processed folder if it does not exist
        Path("data/processed").mkdir(parents=True, exist_ok=True)

        # Save cleaned dataset
        df_clean.to_csv(PROCESSED_PATH, index=False)
        logger.info("Cleaned data saved")

    except Exception as e:
        logger.error(f"make_dataset failed: {e}")
        raise


if __name__ == "__main__":
    main()
