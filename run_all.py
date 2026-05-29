import json
import os
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import plot_tree

# Import modularized components developed to address specific project requirements
from src.clustering import optimal_k_elbow, plot_dendrogram, run_hierarchical, run_kmeans
from src.evaluate import (
    compute_metrics,
    compute_classification_metrics,
    plot_classification_metrics,
    plot_confusion_matrix,
    plot_fit_times,
    plot_regression_metrics,
)
from src.export_rf_tree import export_random_forest_tree
from src.hyperopt import greedy_search, grid_search, optuna_study
from src.models import get_models, get_models_for_svd
from src.pca_svd import apply_pca
from src.preprocessing import build_preprocess, get_preprocess_feature_names
from src.recommender import SimpleSVD, get_top_n_recommendations
from src.train import evaluate_model, train_and_log

# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================
REPORTS_DIR = "reports"     # Directory for generated graphical visualisations
ARTIFACTS_DIR = "artifacts" # Directory for serialized models and JSON logs
EXPERIMENT_NAME = "Projeto_Grupo_8"
SAMPLE_SIZE = 500000        # Downsampling limit to keep execution times reasonable
RECOMMENDER_SAMPLE_SIZE = 20000
RANDOM_STATE = 42           # Fixed seed to ensure reproducible scientific results

# Blacklist specifically slow models from exhaustive hyperparameter optimization
SKIP_OPTIMIZATION = ["SVC_linear", "SVC_rbf", "SVC_poly", "MLPClassifier_single", "MLPClassifier_multi"]
SKIP_REGRESSION_OPTIMIZATION = ["SVR_linear", "SVR_rbf", "SVR_poly", "MLP_single", "MLP_multi"]

# Define explicit data types to optimize memory usage during pandas loading
RATING_DTYPES = {
    "username": "category",
    "anime_id": "int32",
    "status": "category",
    "score": "float32",
    "is_rewatching": "float32",
    "num_watched_episodes": "float32",
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def ensure_dirs():
    """Ensure output directories exist before executing the pipeline."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

def save_json(path, payload):
    """Helper to safely serialize dictionaries to JSON."""
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

def json_safe(value):
    """
    Recursively converts NumPy types to native Python types to prevent 
    serialization errors during JSON dumps (crucial for MLflow logging).
    """
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray, list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)

def make_pipeline(df, target, model):
    """
    Constructs a scikit-learn Pipeline linking the custom preprocessor 
    and the chosen estimator.
    """
    return Pipeline(
        [
            ("preprocess", build_preprocess(df, target)),
            ("model", model),
        ]
    )

def plot_alpha_analysis(df_alpha, model_name, metric_name, output_path):
    """
    Generates a line plot visualizing the impact of the 'alpha' regularization 
    parameter. This fulfills the requirement: 'Ridge and lasso regression 
    (including alpha analysis)'.
    """
    df_alpha = df_alpha.sort_values("alpha")
    best_row = df_alpha.loc[df_alpha[metric_name].idxmin()]

    plt.figure(figsize=(10, 6))
    plt.plot(df_alpha["alpha"], df_alpha[metric_name], marker="o", linewidth=2.2)
    # Highlight the optimal alpha value
    plt.scatter(best_row["alpha"], best_row[metric_name], s=90, zorder=3, label="Best RMSE")
    
    # Add textual annotation for clarity in the final report
    plt.annotate(
        f"best alpha={best_row['alpha']:g}\n{metric_name.upper()}={best_row[metric_name]:.4f}",
        xy=(best_row["alpha"], best_row[metric_name]),
        xytext=(10, 12),
        textcoords="offset points",
        fontsize=9,
    )
    plt.xscale("log")
    plt.title(f"{model_name} alpha analysis")
    plt.xlabel("alpha")
    plt.ylabel(metric_name.upper())
    plt.grid(True, which="both", linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()

def plot_fit_time_comparison(baseline_df, pca_df, output_path):
    """
    Generates a grouped bar chart comparing training times before and after 
    SVD dimensionality reduction. Addresses the requirement: 'compare accuracy 
    and time execution of the previous machine learning models'.
    """
    # Merge the baseline and SVD DataFrames on the 'model' column
    comparison = (
        baseline_df[["model", "fit_time"]]
        .rename(columns={"fit_time": "fit_time_baseline"})
        .merge(
            pca_df[["model", "fit_time"]].rename(columns={"fit_time": "fit_time_svd"}),
            on="model",
        )
        .sort_values("model")
    )

    x = np.arange(len(comparison))
    width = 0.38

    plt.figure(figsize=(12, 6))
    plt.bar(x - width / 2, comparison["fit_time_baseline"], width, label="Baseline")
    plt.bar(x + width / 2, comparison["fit_time_svd"], width, label="SVD")
    plt.xticks(x, comparison["model"], rotation=35, ha="right")
    plt.title("Fit Time Comparison: Baseline vs SVD Reduced Features")
    plt.xlabel("Model")
    plt.ylabel("Fit time (seconds)")
    plt.legend(title="Feature Representation")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def export_gini_tree(model, feature_names, output_path):
    """
    Exports a standalone graphic of a Decision Tree to specifically address 
    the requirement: 'Decision trees. Draw a legible tree and analyse the gini'.
    """
    plt.figure(figsize=(24, 12))
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=["low_score", "high_score"],
        filled=True,
        rounded=True,
        max_depth=3, # Constrained depth ensures the graphic remains interpretable
    )
    plt.title("Decision Tree Classifier (Criterion = Gini Impurity)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()

def run_recommendation(df: pd.DataFrame, artifacts_dir: Path) -> None:
    """
    Executes the custom SimpleSVD recommendation algorithm. Addresses the 
    challenge requirement: 'Employ recommendation algorithms to your dataset'.
    """
    print("\n-- Recommendation Pipeline ---------------------------------------")

    # Isolate relevant columns and clean data
    df_recs = (
        df[["username", "anime_id", "score"]]
        .dropna()
        .drop_duplicates(subset=["username", "anime_id"], keep="last")
        .copy()
    )
    df_recs["score"] = df_recs["score"].astype(float)
    df_recs = df_recs[df_recs["score"] > 0]

    # Downsample if necessary to manage computational overhead
    if len(df_recs) > RECOMMENDER_SAMPLE_SIZE:
        df_recs = df_recs.sample(n=RECOMMENDER_SAMPLE_SIZE, random_state=RANDOM_STATE)

    print(f"  Training SimpleSVD Matrix Factorization on {len(df_recs):,} ratings...")
    svd = SimpleSVD(
        n_factors=50,
        n_epochs=30,
        lr=0.005,
        reg=0.02,
        random_state=RANDOM_STATE,
    )
    
    start = time.time()
    svd.fit(df_recs)
    recommender_fit_time = time.time() - start

    # Define export paths
    artifacts_dir.mkdir(exist_ok=True)
    model_path = artifacts_dir / "svd_recommender.pkl"
    ratings_path = artifacts_dir / "svd_train_ratings.csv"
    anime_ids_path = artifacts_dir / "svd_anime_ids.json"

    # Export serialized models and supporting data for the dashboard
    joblib.dump(svd, model_path)
    df_recs.to_csv(ratings_path, index=False)
    anime_ids_path.write_text(
        json.dumps(df_recs["anime_id"].unique().tolist()),
        encoding="utf-8",
    )

    # Generate an example output (Top 10 Recommendations for the first user)
    example_user = df_recs["username"].iloc[0]
    top_recommendations = get_top_n_recommendations(svd, example_user, df_recs, n=10)
    top_recommendations_df = pd.DataFrame(top_recommendations, columns=["anime_id", "predicted_score"])
    top_recommendations_df.insert(0, "username", example_user)
    
    top_recommendations_path = artifacts_dir / "top_recommendations_example.csv"
    top_recommendations_df.to_csv(top_recommendations_path, index=False)

    recommender_summary_path = artifacts_dir / "recommender_summary.json"
    save_json(
        recommender_summary_path,
        {
            "algorithm": "SimpleSVD matrix factorization",
            "features": ["username", "anime_id"],
            "target": "score",
            "fit_time": recommender_fit_time,
            "train_rows": int(len(df_recs)),
            "example_user": str(example_user),
        },
    )

    # Log all outputs to MLflow
    with mlflow.start_run(run_name="recommendation_svd"):
        mlflow.log_param("algorithm", "SimpleSVD")
        mlflow.log_param("n_factors", svd.n_factors)
        mlflow.log_param("n_epochs", svd.n_epochs)
        mlflow.log_param("recommender_sample_size", int(len(df_recs)))
        mlflow.log_metric("fit_time", recommender_fit_time)
        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(ratings_path))
        mlflow.log_artifact(str(anime_ids_path))
        mlflow.log_artifact(str(top_recommendations_path))
        mlflow.log_artifact(str(recommender_summary_path))

    print("  SVD artifact exported: svd_recommender.pkl")


# =============================================================================
# MAIN EXECUTION PIPELINE
# =============================================================================
def main():
    ensure_dirs()
    mlflow.set_experiment(EXPERIMENT_NAME)

    # 1. Data Ingestion & Cleaning
    full_df = pd.read_csv("datasets/ratings.csv", dtype=RATING_DTYPES)
    full_df = full_df[full_df["score"] > 0]
    full_df = full_df[full_df["num_watched_episodes"] <= 10000] # Outlier removal
    full_df = full_df[full_df["num_watched_episodes"] >= 0]
    
    df = full_df.sample(n=min(SAMPLE_SIZE, len(full_df)), random_state=RANDOM_STATE)
    target = "score"
    X_df = df.drop(columns=[target]).copy()
    y = df[target].copy()

    # Train/Test Split
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    # Initialize Preprocessor
    preprocess = build_preprocess(df, target)
    X_train = preprocess.fit_transform(X_train_df)
    X_test = preprocess.transform(X_test_df)

    # =========================================================================
    # TASK 1: BASELINE REGRESSION MODELS
    # =========================================================================
    models = get_models("regression")

    with mlflow.start_run(run_name="baseline_models"):
        baseline_results = []
        for name, model in models.items():
            fitted, fit_time = train_and_log(name, model, X_train, y_train)
            metrics = evaluate_model(name, fitted, X_test, y_test)
            baseline_results.append({"model": name, "fit_time": fit_time, **metrics})

        # Save and log Baseline metrics
        baseline_df = pd.DataFrame(baseline_results).sort_values("rmse")
        baseline_df.to_csv(os.path.join(ARTIFACTS_DIR, "regression_results.csv"), index=False)
        baseline_path = os.path.join(ARTIFACTS_DIR, "baseline_results.json")
        baseline_df.to_json(baseline_path, orient="records")
        mlflow.log_artifact(baseline_path)

        # Generate Visualizations (RMSE & Fit Times)
        rmse_plot_path = os.path.join(ARTIFACTS_DIR, "rmse_by_model.png")
        plot_regression_metrics(baseline_df, "rmse", "RMSE by Regression Model", rmse_plot_path)
        mlflow.log_artifact(rmse_plot_path)

        fit_time_plot_path = os.path.join(ARTIFACTS_DIR, "fit_time_regression.png")
        plot_fit_times(baseline_df, fit_time_plot_path)
        mlflow.log_artifact(fit_time_plot_path)
        
        mlflow.set_tag("best_model_baseline", min(baseline_results, key=lambda row: row["rmse"])["model"])

    # =========================================================================
    # TASK 2: CROSS-VALIDATION
    # =========================================================================
    with mlflow.start_run(run_name="cross_validation"):
        cv_results = []
        for name, model in models.items():
            pipeline_model = make_pipeline(df, target, model)
            scores = cross_val_score(
                pipeline_model,
                X_df,
                y,
                cv=5,
                scoring="neg_mean_squared_error",
                n_jobs=1,
            )
            rmse_scores = np.sqrt(-scores)
            cv_results.append(
                {
                    "model": name,
                    "cv_rmse_mean": rmse_scores.mean(),
                    "cv_rmse_std": rmse_scores.std(),
                }
            )
            mlflow.log_metric(f"{name}_cv_rmse_mean", rmse_scores.mean())
            mlflow.log_metric(f"{name}_cv_rmse_std", rmse_scores.std())

        cv_path = os.path.join(ARTIFACTS_DIR, "cv_results.csv")
        cv_df = pd.DataFrame(cv_results).sort_values("cv_rmse_mean")
        cv_df.to_csv(cv_path, index=False)
        mlflow.log_artifact(cv_path)

        cv_plot_path = os.path.join(REPORTS_DIR, "cross_validation_rmse.png")
        plot_regression_metrics(
            cv_df.rename(columns={"cv_rmse_mean": "rmse"}),
            "rmse",
            "5-Fold Cross-Validation RMSE",
            cv_plot_path,
        )
        mlflow.log_artifact(cv_plot_path)

    # =========================================================================
    # TASK 3: GRID SEARCH (Ridge, Lasso Alpha Analysis & KNN)
    # =========================================================================
    with mlflow.start_run(run_name="grid_search"):
        alpha_records = []

        # Ridge Regression Alpha Analysis
        ridge_pipeline = make_pipeline(df, target, models["Ridge"])
        ridge_grid = {"model__alpha": [0.01, 0.1, 1, 10, 100]}
        best_ridge = grid_search(ridge_pipeline, ridge_grid, X_train_df, y_train)
        ridge_metrics = evaluate_model("Ridge_grid", best_ridge, X_test_df, y_test)
        
        for alpha in [0.01, 0.1, 1, 10, 100]:
            candidate = make_pipeline(df, target, get_models("regression")["Ridge"].set_params(alpha=alpha))
            start = time.time()
            candidate.fit(X_train_df, y_train)
            fit_time = time.time() - start
            metrics = evaluate_model(f"Ridge_alpha_{alpha}", candidate, X_test_df, y_test)
            alpha_records.append({"model": "Ridge", "alpha": alpha, "fit_time": fit_time, **metrics})

        # Lasso Regression Alpha Analysis
        lasso_pipeline = make_pipeline(df, target, models["Lasso"])
        lasso_grid = {"model__alpha": [0.001, 0.01, 0.1, 1, 10]}
        best_lasso = grid_search(lasso_pipeline, lasso_grid, X_train_df, y_train)
        lasso_metrics = evaluate_model("Lasso_grid", best_lasso, X_test_df, y_test)
        
        for alpha in [0.001, 0.01, 0.1, 1, 10]:
            candidate = make_pipeline(df, target, get_models("regression")["Lasso"].set_params(alpha=alpha))
            start = time.time()
            candidate.fit(X_train_df, y_train)
            fit_time = time.time() - start
            metrics = evaluate_model(f"Lasso_alpha_{alpha}", candidate, X_test_df, y_test)
            alpha_records.append({"model": "Lasso", "alpha": alpha, "fit_time": fit_time, **metrics})

        # KNN Optimization (explaining optimal number of neighbors)
        knn_pipeline = make_pipeline(df, target, models["KNN"])
        knn_grid = {"model__n_neighbors": list(range(3, 32, 2)), "model__weights": ["uniform", "distance"]}
        knn_grid_search = GridSearchCV(
            knn_pipeline,
            knn_grid,
            cv=5,
            scoring="neg_mean_squared_error",
            n_jobs=1,
        )
        knn_grid_search.fit(X_train_df, y_train)
        best_knn = knn_grid_search.best_estimator_
        knn_metrics = evaluate_model("KNN_grid", best_knn, X_test_df, y_test)
        
        # Log KNN Analysis
        knn_analysis = pd.DataFrame(knn_grid_search.cv_results_)
        knn_analysis = knn_analysis.rename(
            columns={
                "param_model__n_neighbors": "n_neighbors",
                "param_model__weights": "weights",
                "mean_test_score": "neg_mse",
                "std_test_score": "neg_mse_std",
            }
        )
        knn_analysis["rmse"] = np.sqrt(-knn_analysis["neg_mse"])
        knn_analysis[["n_neighbors", "weights", "rmse", "neg_mse_std", "rank_test_score"]].to_csv(
            os.path.join(ARTIFACTS_DIR, "knn_analysis.csv"),
            index=False,
        )

        # Log Alpha Analysis Visualisations
        alpha_df = pd.DataFrame(alpha_records)
        alpha_path = os.path.join(ARTIFACTS_DIR, "alpha_analysis.csv")
        alpha_df.to_csv(alpha_path, index=False)
        mlflow.log_artifact(alpha_path)

        ridge_alpha_plot = os.path.join(ARTIFACTS_DIR, "ridge_alpha_analysis.png")
        plot_alpha_analysis(alpha_df[alpha_df["model"] == "Ridge"], "Ridge", "rmse", ridge_alpha_plot)
        mlflow.log_artifact(ridge_alpha_plot)

        lasso_alpha_plot = os.path.join(ARTIFACTS_DIR, "lasso_alpha_analysis.png")
        plot_alpha_analysis(alpha_df[alpha_df["model"] == "Lasso"], "Lasso", "rmse", lasso_alpha_plot)
        mlflow.log_artifact(lasso_alpha_plot)

        knn_summary_path = os.path.join(ARTIFACTS_DIR, "knn_best_result.json")
        save_json(
            knn_summary_path,
            {
                "best_n_neighbors": int(best_knn.named_steps["model"].n_neighbors),
                "metrics": knn_metrics,
                "ridge_metrics": ridge_metrics,
                "lasso_metrics": lasso_metrics,
            },
        )
        mlflow.log_artifact(knn_summary_path)

    best_score_model = None
    best_score_rmse = np.inf

    # =========================================================================
    # TASK 4: GREEDY AND OPTUNA SEARCH (Hyperparameter Optimization)
    # =========================================================================
    with mlflow.start_run(run_name="greedy_search_regression"):
        greedy_regression_results = []
        for name, model in models.items():
            if name in SKIP_REGRESSION_OPTIMIZATION:
                print(f"  Skipping {name} in greedy search to keep runtime manageable.")
                continue
            tuned_model, search_info = greedy_search(
                make_pipeline(df, target, model),
                X_train_df,
                y_train,
                cv=3,
                scoring="neg_mean_squared_error",
            )
            metrics = evaluate_model(f"{name}_greedy", tuned_model, X_test_df, y_test)
            greedy_regression_results.append(
                {
                    "model": name,
                    "search": "greedy",
                    "best_cv_score": search_info["best_score"],
                    "best_params": json.dumps(json_safe(search_info["best_params"])),
                    **metrics,
                }
            )
            # Track the globally best model for final dashboard export
            if metrics["rmse"] < best_score_rmse:
                best_score_rmse = metrics["rmse"]
                best_score_model = tuned_model

        greedy_regression_df = pd.DataFrame(greedy_regression_results).sort_values("rmse")
        greedy_regression_path = os.path.join(ARTIFACTS_DIR, "greedy_regression_results.csv")
        greedy_regression_df.to_csv(greedy_regression_path, index=False)
        mlflow.log_artifact(greedy_regression_path)

    with mlflow.start_run(run_name="optuna_search_regression"):
        optuna_regression_results = []
        for name, model in models.items():
            if name in SKIP_REGRESSION_OPTIMIZATION:
                print(f"  Skipping {name} in Optuna to keep runtime manageable.")
                continue
            tuned_model, search_info = optuna_study(
                make_pipeline(df, target, model),
                X_train_df,
                y_train,
                n_trials=10,
                cv=3,
                scoring="neg_mean_squared_error",
            )
            metrics = evaluate_model(f"{name}_optuna", tuned_model, X_test_df, y_test)
            optuna_regression_results.append(
                {
                    "model": name,
                    "search": "optuna",
                    "best_cv_score": search_info["best_score"],
                    "best_params": json.dumps(json_safe(search_info["best_params"])),
                    **metrics,
                }
            )
            if metrics["rmse"] < best_score_rmse:
                best_score_rmse = metrics["rmse"]
                best_score_model = tuned_model

        optuna_regression_df = pd.DataFrame(optuna_regression_results).sort_values("rmse")
        optuna_regression_path = os.path.join(ARTIFACTS_DIR, "optuna_regression_results.csv")
        optuna_regression_df.to_csv(optuna_regression_path, index=False)
        mlflow.log_artifact(optuna_regression_path)

    # Fallback mechanism
    if best_score_model is None:
        fallback_name = baseline_df.iloc[0]["model"]
        best_score_model = make_pipeline(df, target, models[fallback_name])
        best_score_model.fit(X_train_df, y_train)

    # =========================================================================
    # TASK 5: DIMENSIONALITY REDUCTION (SVD/PCA)
    # =========================================================================
    # Reduces feature space while retaining 95% variance
    X_train_red, svd = apply_pca(X_train, variance=0.95)
    X_test_red = svd.transform(X_test)
    models_svd = get_models_for_svd("regression")

    with mlflow.start_run(run_name="pca_models"):
        pca_results = []
        for name, model in models_svd.items():
            fitted, fit_time = train_and_log(f"{name}_svd", model, X_train_red, y_train)
            metrics = evaluate_model(f"{name}_svd", fitted, X_test_red, y_test)
            
            # Flag unstable models (e.g., polynomial kernels failing on dense SVD data)
            unstable = not np.isfinite(metrics["rmse"]) or metrics["rmse"] > 10
            pca_results.append({"model": name, "fit_time": fit_time, "svd_unstable": unstable, **metrics})

        pca_df = pd.DataFrame(pca_results).sort_values("rmse")
        pca_plot_df = pca_df[~pca_df["svd_unstable"]].copy()
        
        pca_path = os.path.join(ARTIFACTS_DIR, "pca_results.json")
        pca_df.to_json(pca_path, orient="records")
        mlflow.log_artifact(pca_path)

        # Plot PCA metrics vs Baseline
        pca_rmse_plot_path = os.path.join(REPORTS_DIR, "pca_rmse.png")
        plot_regression_metrics(pca_plot_df, "rmse", "RMSE after SVD dimensionality reduction", pca_rmse_plot_path)
        mlflow.log_artifact(pca_rmse_plot_path)

        pca_fit_time_plot_path = os.path.join(REPORTS_DIR, "pca_fit_times.png")
        plot_fit_times(pca_plot_df, pca_fit_time_plot_path)
        mlflow.log_artifact(pca_fit_time_plot_path)

        fit_time_comparison_path = os.path.join(REPORTS_DIR, "fit_time_comparison_baseline_vs_svd.png")
        plot_fit_time_comparison(baseline_df, pca_plot_df, fit_time_comparison_path)
        mlflow.log_artifact(fit_time_comparison_path)

        pca_summary_path = os.path.join(ARTIFACTS_DIR, "pca_summary.json")
        save_json(
            pca_summary_path,
            {
                "target_variance": 0.95,
                "selected_components": int(svd.n_components),
                "explained_variance_sum": float(np.sum(svd.explained_variance_ratio_)),
            },
        )
        mlflow.log_artifact(pca_summary_path)
        mlflow.set_tag("best_model_pca", min(pca_results, key=lambda row: row["rmse"])["model"])
        
        # Generate Delta (Difference) DataFrame for paper analysis
        pca_comparison = (
            baseline_df[["model", "fit_time", "rmse"]]
            .rename(columns={"fit_time": "fit_time_baseline", "rmse": "rmse_baseline"})
            .merge(
                pca_df[["model", "fit_time", "rmse"]].rename(
                    columns={"fit_time": "fit_time_svd", "rmse": "rmse_svd"}
                ),
                on="model",
            )
        )
        pca_comparison["rmse_delta"] = pca_comparison["rmse_svd"] - pca_comparison["rmse_baseline"]
        pca_comparison["fit_time_delta"] = pca_comparison["fit_time_svd"] - pca_comparison["fit_time_baseline"]
        pca_comparison.to_csv(os.path.join(ARTIFACTS_DIR, "pca_comparison.csv"), index=False)

    # =========================================================================
    # TASK 6: CLASSIFICATION PIPELINE
    # =========================================================================
    # Convert numerical target to binary classification (Score >= 7)
    df_class = df.copy()
    df_class["high_score"] = (df_class["score"] >= 7).astype(int)
    class_target = "high_score"
    X_class_df = df_class.drop(columns=["score", class_target]).copy()
    y_class = df_class[class_target].copy()

    # Stratified split to maintain class balance
    Xc_train_df, Xc_test_df, yc_train, yc_test = train_test_split(
        X_class_df,
        y_class,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_class,
    )

    preprocess_class = build_preprocess(df_class.drop(columns=["score"]), class_target)
    Xc_train = preprocess_class.fit_transform(Xc_train_df)
    Xc_test = preprocess_class.transform(Xc_test_df)
    class_feature_names = get_preprocess_feature_names(preprocess_class).tolist()

    clf_models = get_models("classification")
    with mlflow.start_run(run_name="classification_models"):
        
        # GridSearch specific to KNN Classifier
        knn_clf_grid = {"n_neighbors": list(range(3, 32, 2)), "weights": ["uniform", "distance"]}
        best_knn_clf = grid_search(
            clf_models["KNNClassifier"],
            knn_clf_grid,
            Xc_train,
            yc_train,
            scoring="f1",
        )
        clf_models["KNNClassifier"] = best_knn_clf

        best_knn_clf_preds = best_knn_clf.predict(Xc_test)
        knn_clf_metrics = compute_classification_metrics(yc_test, best_knn_clf_preds)
        knn_clf_summary_path = os.path.join(ARTIFACTS_DIR, "knn_classifier_best_result.json")
        save_json(
            knn_clf_summary_path,
            {
                "best_n_neighbors": int(best_knn_clf.n_neighbors),
                "best_weights": best_knn_clf.weights,
                "metrics": json_safe(knn_clf_metrics),
            },
        )
        mlflow.log_artifact(knn_clf_summary_path)

        # Train and Evaluate all Classification Models
        class_results = []
        for name, model in clf_models.items():
            start = time.time()
            model.fit(Xc_train, yc_train)
            fit_time = time.time() - start
            preds = model.predict(Xc_test)

            metrics = {
                "model": name,
                "fit_time": fit_time,
                **compute_classification_metrics(yc_test, preds),
            }
            class_results.append(metrics)

            # Log metrics and artifacts (Confusion Matrices & Trees)
            for metric_name, metric_value in metrics.items():
                if metric_name != "model":
                    mlflow.log_metric(f"{name}_{metric_name}", metric_value)

            mlflow.sklearn.log_model(model, f"{name}_model")

            cm_path = os.path.join(REPORTS_DIR, f"{name.lower()}_confusion_matrix.png")
            plot_confusion_matrix(yc_test, preds, f"Confusion Matrix: {name}", cm_path)
            mlflow.log_artifact(cm_path)

            if name == "DecisionTreeClassifier":
                gini_tree_path = os.path.join(ARTIFACTS_DIR, "decision_tree.png")
                export_gini_tree(model, class_feature_names, gini_tree_path)
                mlflow.log_artifact(gini_tree_path)

                gini_summary_path = os.path.join(ARTIFACTS_DIR, "decision_tree_info.json")
                save_json(
                    gini_summary_path,
                    {
                        "criterion": model.criterion,
                        "depth": int(model.get_depth()),
                        "leaves": int(model.get_n_leaves()),
                        "root_gini": float(model.tree_.impurity[0]),
                    },
                )
                mlflow.log_artifact(gini_summary_path)

        class_df = pd.DataFrame(class_results).sort_values("f1", ascending=False)
        class_df.to_csv(os.path.join(ARTIFACTS_DIR, "classification_results.csv"), index=False)
        class_path = os.path.join(ARTIFACTS_DIR, "classification_results.json")
        class_df.to_json(class_path, orient="records")
        mlflow.log_artifact(class_path)

        class_plot_path = os.path.join(ARTIFACTS_DIR, "classification_metrics.png")
        plot_classification_metrics(class_df, class_plot_path)
        mlflow.log_artifact(class_plot_path)

        class_fit_time_plot_path = os.path.join(ARTIFACTS_DIR, "fit_time_classification.png")
        plot_fit_times(class_df, class_fit_time_plot_path)
        mlflow.log_artifact(class_fit_time_plot_path)
        
        # Identify Best Classification Model
        best_class_name = class_df.iloc[0]["model"]
        best_class_preds = clf_models[best_class_name].predict(Xc_test)
        confusion_matrix_path = os.path.join(ARTIFACTS_DIR, "confusion_matrix.png")
        plot_confusion_matrix(yc_test, best_class_preds, f"Confusion Matrix: {best_class_name}", confusion_matrix_path)
        mlflow.log_artifact(confusion_matrix_path)
        mlflow.set_tag("best_model_classification", class_df.iloc[0]["model"])

    best_classifier_model = clf_models[class_df.iloc[0]["model"]]
    best_classifier_f1 = float(class_df.iloc[0]["f1"])

    # Greedy & Optuna Optimization for Classification
    with mlflow.start_run(run_name="greedy_search_classification"):
        greedy_classification_results = []
        for name, model in get_models("classification").items():
            if name in SKIP_OPTIMIZATION:
                print(f"Skipping {name} - too slow for optimization")
                continue
            tuned_model, search_info = greedy_search(
                Pipeline(
                    [
                        ("preprocess", build_preprocess(df_class.drop(columns=["score"]), class_target)),
                        ("model", model),
                    ]
                ),
                X_class_df,
                y_class,
                cv=3,
                scoring="f1",
            )
            preds = tuned_model.predict(Xc_test_df)
            metrics = compute_classification_metrics(yc_test, preds)
            greedy_classification_results.append(
                {
                    "model": name,
                    "search": "greedy",
                    "best_cv_score": search_info["best_score"],
                    "best_params": json.dumps(json_safe(search_info["best_params"])),
                    **metrics,
                }
            )
            if metrics["f1"] > best_classifier_f1:
                best_classifier_f1 = metrics["f1"]
                best_classifier_model = tuned_model

        greedy_classification_df = pd.DataFrame(greedy_classification_results).sort_values("f1", ascending=False)
        greedy_classification_path = os.path.join(ARTIFACTS_DIR, "greedy_classification_results.csv")
        greedy_classification_df.to_csv(greedy_classification_path, index=False)
        mlflow.log_artifact(greedy_classification_path)

    with mlflow.start_run(run_name="optuna_search_classification"):
        optuna_classification_results = []
        for name, model in get_models("classification").items():
            if name in SKIP_OPTIMIZATION:
                print(f"Skipping {name} - too slow for optimization")
                continue
            tuned_model, search_info = optuna_study(
                Pipeline(
                    [
                        ("preprocess", build_preprocess(df_class.drop(columns=["score"]), class_target)),
                        ("model", model),
                    ]
                ),
                X_class_df,
                y_class,
                n_trials=10,
                cv=3,
                scoring="f1",
            )
            preds = tuned_model.predict(Xc_test_df)
            metrics = compute_classification_metrics(yc_test, preds)
            optuna_classification_results.append(
                {
                    "model": name,
                    "search": "optuna",
                    "best_cv_score": search_info["best_score"],
                    "best_params": json.dumps(json_safe(search_info["best_params"])),
                    **metrics,
                }
            )
            if metrics["f1"] > best_classifier_f1:
                best_classifier_f1 = metrics["f1"]
                best_classifier_model = tuned_model

        optuna_classification_df = pd.DataFrame(optuna_classification_results).sort_values("f1", ascending=False)
        optuna_classification_path = os.path.join(ARTIFACTS_DIR, "optuna_classification_results.csv")
        optuna_classification_df.to_csv(optuna_classification_path, index=False)
        mlflow.log_artifact(optuna_classification_path)

    with mlflow.start_run(run_name="cross_validation_classification"):
        cv_class_results = []
        clf_models_cv = get_models("classification")
        for name, model in clf_models_cv.items():
            pipeline_model = Pipeline(
                [
                    ("preprocess", build_preprocess(df_class.drop(columns=["score"]), class_target)),
                    ("model", model),
                ]
            )
            scores = cross_val_score(
                pipeline_model,
                X_class_df,
                y_class,
                cv=5,
                scoring="f1",
                n_jobs=1,
            )
            cv_class_results.append(
                {
                    "model": name,
                    "cv_f1_mean": scores.mean(),
                    "cv_f1_std": scores.std(),
                }
            )
            mlflow.log_metric(f"{name}_cv_f1_mean", scores.mean())
            mlflow.log_metric(f"{name}_cv_f1_std", scores.std())

        cv_class_path = os.path.join(ARTIFACTS_DIR, "cv_classification_results.csv")
        cv_class_df = pd.DataFrame(cv_class_results).sort_values("cv_f1_mean", ascending=False)
        cv_class_df.to_csv(cv_class_path, index=False)
        mlflow.log_artifact(cv_class_path)

        cv_class_plot_path = os.path.join(REPORTS_DIR, "cross_validation_classification_f1.png")
        plot_regression_metrics(
            cv_class_df.rename(columns={"cv_f1_mean": "f1"}),
            "f1",
            "5-Fold Cross-Validation F1 (Classification)",
            cv_class_plot_path,
        )
        mlflow.log_artifact(cv_class_plot_path)

    # =========================================================================
    # TASK 7: CLUSTERING ANALYSIS
    # =========================================================================
    # Apply clustering on the reduced feature space for computational efficiency
    cluster_input = X_train_red[:10000]
    with mlflow.start_run(run_name="clustering_analysis"):
        
        # Determine optimal K using Elbow Method
        elbow_plot_path = os.path.join(ARTIFACTS_DIR, "elbow_kmeans.png")
        cluster_results, best_k = optimal_k_elbow(cluster_input, max_k=10, output_path=elbow_plot_path)
        
        clustering_metrics_path = os.path.join(ARTIFACTS_DIR, "cluster_results.csv")
        cluster_results.to_csv(clustering_metrics_path, index=False)
        mlflow.log_artifact(clustering_metrics_path)
        mlflow.log_artifact(elbow_plot_path)

        # Run KMeans and Hierarchical Clustering
        labels_km, inertia = run_kmeans(cluster_input, n_clusters=best_k)
        labels_hc = run_hierarchical(cluster_input, n_clusters=best_k, linkage_method="ward")
        
        # Plot Dendrogram
        dendrogram_path = os.path.join(ARTIFACTS_DIR, "dendrogram.png")
        plot_dendrogram(cluster_input, method="ward", sample_size=1000, output_path=dendrogram_path)
        mlflow.log_artifact(dendrogram_path)

        cluster_assignments = pd.DataFrame({"km_cluster": labels_km, "hc_cluster": labels_hc})
        cluster_assignments_path = os.path.join(ARTIFACTS_DIR, "cluster_assignments.csv")
        cluster_assignments.to_csv(cluster_assignments_path, index=False)
        mlflow.log_artifact(cluster_assignments_path)

        cluster_summary_path = os.path.join(ARTIFACTS_DIR, "clustering_summary.json")
        save_json(cluster_summary_path, {"selected_k": int(best_k), "kmeans_inertia": float(inertia)})
        mlflow.log_artifact(cluster_summary_path)

    # =========================================================================
    # TASK 8: RECOMMENDATION SYSTEM
    # =========================================================================
    run_recommendation(df, Path(ARTIFACTS_DIR))

    # =========================================================================
    # FINAL EXPORTS & ARTIFACT MANAGEMENT
    # =========================================================================
    joblib.dump(preprocess, os.path.join(ARTIFACTS_DIR, "preprocess.pkl"))
    joblib.dump(preprocess_class, os.path.join(ARTIFACTS_DIR, "preprocess_classification.pkl"))

    if not isinstance(best_score_model, Pipeline) or "preprocess" not in best_score_model.named_steps:
        base_model = best_score_model.named_steps["model"] if isinstance(best_score_model, Pipeline) else best_score_model
        best_score_model = make_pipeline(df, target, base_model)
        best_score_model.fit(X_train_df, y_train)

    joblib.dump(best_score_model, os.path.join(ARTIFACTS_DIR, "best_score_model.pkl"))
    joblib.dump(best_classifier_model, os.path.join(ARTIFACTS_DIR, "best_classification_model.pkl"))

    feature_columns_path = os.path.join(ARTIFACTS_DIR, "training_feature_columns.json")
    save_json(
        feature_columns_path,
        {
            "regression_features": X_df.columns.tolist(),
            "classification_features": X_class_df.columns.tolist(),
        },
    )

    with mlflow.start_run(run_name="model_artifacts"):
        mlflow.log_artifact(os.path.join(ARTIFACTS_DIR, "preprocess.pkl"))
        mlflow.log_artifact(os.path.join(ARTIFACTS_DIR, "preprocess_classification.pkl"))
        mlflow.log_artifact(os.path.join(ARTIFACTS_DIR, "best_score_model.pkl"))
        mlflow.log_artifact(os.path.join(ARTIFACTS_DIR, "best_classification_model.pkl"))
        mlflow.log_artifact(feature_columns_path)

        # Export representative Random Forest Tree for the paper
        try:
            rf_model = best_score_model.named_steps["model"]
            rf_feature_names = get_preprocess_feature_names(best_score_model.named_steps["preprocess"])
            export_random_forest_tree(
                rf_model,
                os.path.join(ARTIFACTS_DIR, "rf_tree.dot"),
                os.path.join(ARTIFACTS_DIR, "rf_tree.png"),
                feature_names=rf_feature_names,
            )
            mlflow.log_artifact(os.path.join(ARTIFACTS_DIR, "rf_tree.png"))
        except Exception as exc:
            error_path = os.path.join(ARTIFACTS_DIR, "rf_tree_export_error.json")
            save_json(error_path, {"error": str(exc)})
            mlflow.log_artifact(error_path)


if __name__ == "__main__":
    main()
