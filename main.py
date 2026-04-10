import logging
from src.data.make_dataset import main as make_dataset
from src.features.build_features import build_features
from src.models.train_model import train_model, evaluate_model, save_model
from src.visualization.visualize import plot_feature_importance

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_pipeline():
    try:
        logger.info("Pipeline started")

        # Step 1 — Clean raw dataset
        make_dataset()

        # Step 2 — Build features
        df, X, y = build_features()

        # Step 3 — Train model
        model, X_test, y_test = train_model(X, y)

        # Step 4 — Evaluate model
        mae, r2 = evaluate_model(model, X_test, y_test)

        logger.info(f"MAE: {mae:.2f}")
        logger.info(f"R2: {r2:.4f}")

        # Step 5 — Save model
        save_model(model)

        # Step 6 — Visualize feature importance
        plot_feature_importance(model, X)

        logger.info("Pipeline finished")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")


if __name__ == "__main__":
    run_pipeline()
