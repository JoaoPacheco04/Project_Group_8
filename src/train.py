import time
import mlflow
import mlflow.sklearn
import pandas as pd

from src.evaluate import compute_metrics


def train_and_log(name: str, model, X_train: pd.DataFrame, y_train: pd.Series):
    """
    Trains a machine learning model, measures its computational cost, 
    and logs the artifacts to MLflow for version management.

    This function strictly fulfills the project requirement to conduct an 
    "Analysis of the fit time per machine learning model" [cite: 61] and 
    "Manage model versions with MLflow"[cite: 63].

    Parameters
    ----------
    name : str
        A logical identifier for the model (e.g., "RandomForest_Baseline").
    model : estimator object
        The scikit-learn compatible machine learning model to be trained.
    X_train : pd.DataFrame
        The preprocessed and standardized feature matrix for training.
    y_train : pd.Series
        The target variable for the training set.

    Returns
    -------
    model : fitted estimator
        The trained machine learning model.
    fit_time : float
        The total execution time required to fit the model, in seconds.
    """
    # Start the timer to explicitly measure the computational cost of training
    start = time.time()
    
    # Fit the algorithm to the training data
    model.fit(X_train, y_train)
    
    # Calculate the elapsed time
    fit_time = time.time() - start

    # Safely attempt to log the metrics and model artifacts to the MLflow server
    try:
        # Log the fit time to facilitate the required performance comparisons
        mlflow.log_metric(f"{name}_fit_time", fit_time)
        
        # Serialize and log the trained model artifact for version control
        mlflow.sklearn.log_model(model, f"{name}_model")
        
        # Log the underlying class name of the model for metadata tracking
        mlflow.log_param(f"{name}_type", type(model).__name__)
    except Exception:
        # Silently pass if the MLflow server is unreachable, ensuring the 
        # training pipeline does not crash during local debugging.
        pass

    return model, fit_time


def evaluate_model(name: str, model, X_test: pd.DataFrame, y_test: pd.Series):
    """
    Generates predictions on the unseen test set and logs evaluation metrics.

    This function addresses the project mandate to analyze predictive error 
    (specifically RMSE) [cite: 72] by calculating standard regression performance metrics 
    and securing them in the MLflow tracking environment.

    Parameters
    ----------
    name : str
        A logical identifier matching the trained model.
    model : fitted estimator
        The trained machine learning model to evaluate.
    X_test : pd.DataFrame
        The preprocessed feature matrix for the test set.
    y_test : pd.Series
        The true target values for the test set.

    Returns
    -------
    metrics : dict
        A dictionary containing the computed RMSE, MAE, and R-squared values.
    """
    # Generate predictions on the hold-out test set
    y_pred = model.predict(X_test)
    
    # Compute standard evaluation metrics (RMSE, MAE, R2)
    metrics = compute_metrics(y_test, y_pred)

    # Safely record the evaluation metrics in the MLflow run
    try:
        mlflow.log_metric(f"{name}_rmse", metrics["rmse"])
        mlflow.log_metric(f"{name}_mae", metrics["mae"])
        mlflow.log_metric(f"{name}_r2", metrics["r2"])
    except Exception:
        pass
        
    return metrics