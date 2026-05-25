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
    try:
        return mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:
        return mean_squared_error(y_true, y_pred) ** 0.5


def compute_metrics(y_true: pd.Series, y_pred: pd.Series):
    """Return a dict with RMSE, MAE and R2."""
    rmse = _rmse(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"rmse": rmse, "mae": mae, "r2": r2}


def compute_classification_metrics(y_true: pd.Series, y_pred: pd.Series):
    """Return standard binary classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def plot_metrics(df_metrics: pd.DataFrame, metric: str, title: str = None, output_path: str = None):
    """Bar plot of a given metric (rmse, mae, r2) across models.
    ``df_metrics`` must have a column ``model`` and the metric name.
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_metrics, x="model", y=metric)
    plt.title(title or f"Comparison of {metric.upper()} across models")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=180)
        plt.close()
    else:
        plt.show()


def plot_regression_metrics(df_metrics: pd.DataFrame, metric_name: str, title: str, output_path: str):
    """Save a sorted bar plot for a regression metric."""
    ordered = df_metrics.sort_values(metric_name)
    plot_metrics(ordered, metric_name, title, output_path)


def plot_fit_times(df_times: pd.DataFrame, output_path: str = None, title: str = "Fit time per model"):
    """Bar plot of fit times (seconds) for each model."""
    plt.figure(figsize=(12, 6))
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
    """Save a grouped bar plot for accuracy, precision, recall and F1."""
    melted = df_metrics.melt(
        id_vars=["model"],
        value_vars=["accuracy", "precision", "recall", "f1"],
        var_name="metric",
        value_name="score",
    )
    plt.figure(figsize=(10, 6))
    sns.barplot(data=melted, x="model", y="score", hue="metric")
    plt.title("Classification metrics by model")
    plt.ylim(0, 1)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()


def plot_confusion_matrix(y_true, y_pred, title: str, output_path: str):
    """Save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
