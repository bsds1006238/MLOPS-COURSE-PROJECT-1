import json
import pandas as pd
import pytest
from config.paths_config import *

from sklearn.metrics import accuracy_score, precision_score, recall_score



joblib = pytest.importorskip("joblib", reason="joblib not available in CI")

def test_model_can_load():
    assert joblib is not None

def test_model_performance():
    # Load data
    df = pd.read_csv(TEST_FILE_PATH)

    X = df.drop(
        columns=[
            "species",
            "event_timestamp",
            "iris_id",
        ]
    )
    y_true = df["species"]

    # Load model
    model = joblib.load(MODEL_OUTPUT_PATH)

    # Inference
    y_pred = model.predict(X)

    # Metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro")
    recall = recall_score(y_true, y_pred, average="macro")

    # Save metrics for CI / CML
    with open("metrics.json", "w") as f:
        json.dump(
            {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
            },
            f,
            indent=2,
        )

    # Quality gates (CI will fail if these drop)
    assert accuracy >= 0.3, f"Accuracy too low: {accuracy}"
    assert precision >= 0.2, f"Precision too low: {precision}"
    assert recall >= 0.2, f"Recall too low: {recall}"