from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_model(X, y):
    try:
        logger.info("Training model")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestRegressor(
            n_estimators=200,
            criterion="absolute_error",
            random_state=42,
        )

        model.fit(X_train, y_train)

        return model, X_test, y_test

    except Exception as e:
        logger.error(f"train_model error: {e}")
        raise


def evaluate_model(model, X_test, y_test):
    try:
        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        return mae, r2

    except Exception as e:
        logger.error(f"evaluate_model error: {e}")


def save_model(model):
    try:
        os.makedirs("models", exist_ok=True)

        with open("models/RFmodel.pkl", "wb") as f:
            pickle.dump(model, f)

        logger.info("Model saved")

    except Exception as e:
        logger.error(f"save_model error: {e}")
        raise
