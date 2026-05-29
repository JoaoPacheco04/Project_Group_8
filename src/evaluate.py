import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)


def _rmse(y_true, y_pred):
    """
    Helper function to compute the Root Mean Squared Error (RMSE).
    It handles compatibility across different versions of scikit-learn
    (handling the deprecation of the 'squared' parameter).
    """
    try:
        return mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:
        # Fallback for newer scikit-learn versions where 'squared' is removed
        return mean_squared_error(y_true, y_pred) ** 0.5


def compute_metrics(y_true: pd.Series, y_pred: pd.Series):
    """
    Calculates standard regression performance metrics.
    Returns a dictionary containing RMSE, MAE, and R-squared (R2).
    This fulfills the project requirement to evaluate model error (e.g., RMSE).
    """
    rmse = _rmse(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"rmse": rmse, "mae": mae, "r2": r2}


def compute_classification_metrics(y_true: pd.Series, y_pred: pd.Series):
    """
    Calculates standard binary classification metrics.
    Returns Accuracy, Precision, Recall, and F1-score.
    This explicitly fulfills the requirement to analyze Precision, Recall, or F-measure.
    Zero_division=0 prevents warnings when a model fails to predict a specific class.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def plot_metrics(df_metrics: pd.DataFrame, metric: str, title: str = None, output_path: str = None):
    """
    Generates a bar plot comparing a specific metric (e.g., RMSE, MAE) across different models.
    `df_metrics` must contain a 'model' column and a column corresponding to the metric name.
    Useful for visually comparing model performance in the final dashboard or paper.
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_metrics, x="model", y=metric)
    plt.title(title or f"Comparison of {metric.upper()} across models")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    # Save the figure for the scientific paper or display it dynamically
    if output_path:
        plt.savefig(output_path, dpi=180)
        plt.close()
    else:
        plt.show()


def plot_regression_metrics(df_metrics: pd.DataFrame, metric_name: str, title: str, output_path: str):
    """
    Creates and saves a sorted bar plot for regression metrics.
    Sorting the values makes it immediately clear which model performs best.
    """
    ordered = df_metrics.sort_values(metric_name)
    plot_metrics(ordered, metric_name, title, output_path)


def plot_fit_times(df_times: pd.DataFrame, output_path: str = None, title: str = "Fit time per model"):
    """
    Generates a bar plot comparing the training (fit) times of each machine learning model.
    This explicitly satisfies the project requirement to analyze fit times per model.
    """
    plt.figure(figsize=(12, 6))
    # Sort descending so the most computationally expensive models appear first
    ordered = df_times.sort_values("fit_time", ascending=False)
    
    sns.barplot(data=ordered, x="model", y="fit_time")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Fit time (seconds)")
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=180)
        plt.close()
    else:
        plt.show()


def plot_classification_metrics(df_metrics: pd.DataFrame, output_path: str = None):
    """
    Generates a grouped bar chart to simultaneously compare Accuracy, Precision, Recall, and F1-score across models.
    Provides a comprehensive visual analysis required for the final dashboard.
    """
    # Restructure the DataFrame for seaborn's hue grouping
    melted = df_metrics.melt(
        id_vars=["model"],
        value_vars=["accuracy", "precision", "recall", "f1"],
        var_name="metric",
        value_name="score",
    )
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=melted, x="model", y="score", hue="metric")
    plt.title("Classification metrics by model")
    # Standardize the Y-axis from 0 to 1 since these are percentage-based metrics
    plt.ylim(0, 1)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()


def plot_confusion_matrix(y_true, y_pred, title: str, output_path: str):
    """
    Generates and saves a Confusion Matrix heatmap.
    This provides deeper insight into False Positives and False Negatives, 
    enriching the critical analysis section of the scientific paper.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 4))
    
    # Use seaborn heatmap for a cleaner, publication-ready visual
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    
    plt.savefig(output_path)
    plt.close()