import matplotlib.pyplot as plt
import numpy as np
import os


def plot_feature_importance(model, X):

    # ensure models folder exists
    os.makedirs("models", exist_ok=True)

    importances = model.feature_importances_
    indices = np.argsort(importances)

    plt.figure(figsize=(6, 4))

    plt.barh(range(len(indices)), importances[indices])
    plt.yticks(range(len(indices)), X.columns[indices])
    plt.xlabel("Importance")
    plt.title("Feature Importance")

    plt.tight_layout()
    plt.savefig("models/feature_importance.png")

    plt.close()
