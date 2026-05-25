import time
import mlflow
import mlflow.sklearn
import pandas as pd

from src.evaluate import compute_metrics


def train_and_log(name: str, model, X_train: pd.DataFrame, y_train: pd.Series):
    """Fit ``model`` on the training data, measure fit time and log to MLflow.

    Returns
    -------
    model : fitted estimator
    fit_time : float (seconds)
    """
    start = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - start

    # Log artefacts and metrics
    mlflow.log_metric(f"{name}_fit_time", fit_time)
    mlflow.sklearn.log_model(model, f"{name}_model")
    mlflow.log_param(f"{name}_type", type(model).__name__)

    return model, fit_time


def evaluate_model(name: str, model, X_test: pd.DataFrame, y_test: pd.Series):
    """Predict on test set and log RMSE, MAE and R2 to MLflow."""
    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)

    mlflow.log_metric(f"{name}_rmse", metrics["rmse"])
    mlflow.log_metric(f"{name}_mae", metrics["mae"])
    mlflow.log_metric(f"{name}_r2", metrics["r2"])
    return metrics
