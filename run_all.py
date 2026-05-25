import json
import os
import time

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.tree import plot_tree

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
from src.hyperopt import grid_search, optuna_study
from src.models import get_models
from src.pca_svd import apply_pca
from src.preprocessing import build_preprocess
from src.recommender import SimpleSVD, get_top_n_recommendations
from src.train import evaluate_model, train_and_log


REPORTS_DIR = "reports"
ARTIFACTS_DIR = "artifacts"
EXPERIMENT_NAME = "Projeto_Grupo_8"
SAMPLE_SIZE = 500000
RECOMMENDER_SAMPLE_SIZE = 20000


def ensure_dirs():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def json_safe(value):
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
    return Pipeline(
        [
            ("preprocess", build_preprocess(df, target)),
            ("model", model),
        ]
    )


def plot_alpha_analysis(df_alpha, model_name, metric_name, output_path):
    plt.figure(figsize=(8, 5))
    plt.plot(df_alpha["alpha"], df_alpha[metric_name], marker="o")
    plt.xscale("log")
    plt.title(f"{model_name} alpha analysis")
    plt.xlabel("alpha")
    plt.ylabel(metric_name.upper())
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_fit_time_comparison(baseline_df, pca_df, output_path):
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
    plt.title("Fit time comparison: baseline vs SVD")
    plt.xlabel("Model")
    plt.ylabel("Fit time (seconds)")
    plt.legend(title="Feature representation")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def export_gini_tree(model, feature_names, output_path):
    plt.figure(figsize=(24, 12))
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=["low_score", "high_score"],
        filled=True,
        rounded=True,
        max_depth=3,
    )
    plt.title("Decision Tree Classifier (criterion = gini)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main():
    ensure_dirs()
    mlflow.set_experiment(EXPERIMENT_NAME)

    full_df = pd.read_csv("datasets/ratings.csv")
    df = full_df.sample(n=min(SAMPLE_SIZE, len(full_df)), random_state=42)
    target = "score"
    X_df = df.drop(columns=[target]).copy()
    y = df[target].copy()

    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df,
        y,
        test_size=0.2,
        random_state=42,
    )

    preprocess = build_preprocess(df, target)
    X_train = preprocess.fit_transform(X_train_df)
    X_test = preprocess.transform(X_test_df)

    models = get_models("regression")

    with mlflow.start_run(run_name="baseline_models"):
        baseline_results = []
        for name, model in models.items():
            fitted, fit_time = train_and_log(name, model, X_train, y_train)
            metrics = evaluate_model(name, fitted, X_test, y_test)
            baseline_results.append({"model": name, "fit_time": fit_time, **metrics})

        baseline_df = pd.DataFrame(baseline_results).sort_values("rmse")
        baseline_path = os.path.join(ARTIFACTS_DIR, "baseline_results.json")
        baseline_df.to_json(baseline_path, orient="records")
        mlflow.log_artifact(baseline_path)

        rmse_plot_path = os.path.join(REPORTS_DIR, "baseline_rmse.png")
        plot_regression_metrics(baseline_df, "rmse", "RMSE by regression model", rmse_plot_path)
        mlflow.log_artifact(rmse_plot_path)

        fit_time_plot_path = os.path.join(REPORTS_DIR, "baseline_fit_times.png")
        plot_fit_times(baseline_df, fit_time_plot_path)
        mlflow.log_artifact(fit_time_plot_path)
        mlflow.set_tag("best_model_baseline", min(baseline_results, key=lambda row: row["rmse"])["model"])

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
            "5-fold cross-validation RMSE",
            cv_plot_path,
        )
        mlflow.log_artifact(cv_plot_path)

    with mlflow.start_run(run_name="grid_search"):
        alpha_records = []

        ridge_pipeline = make_pipeline(df, target, models["Ridge"])
        ridge_grid = {"model__alpha": [0.01, 0.1, 1, 10, 100]}
        best_ridge = grid_search(ridge_pipeline, ridge_grid, X_train_df, y_train)
        ridge_metrics = evaluate_model("Ridge_grid", best_ridge, X_test_df, y_test)
        for alpha in [0.01, 0.1, 1, 10, 100]:
            candidate = make_pipeline(df, target, get_models("regression")["Ridge"].set_params(alpha=alpha))
            candidate.fit(X_train_df, y_train)
            alpha_records.append(
                {
                    "model": "Ridge",
                    "alpha": alpha,
                    "rmse": evaluate_model(f"Ridge_alpha_{alpha}", candidate, X_test_df, y_test)["rmse"],
                }
            )

        lasso_pipeline = make_pipeline(df, target, models["Lasso"])
        lasso_grid = {"model__alpha": [0.001, 0.01, 0.1, 1, 10]}
        best_lasso = grid_search(lasso_pipeline, lasso_grid, X_train_df, y_train)
        lasso_metrics = evaluate_model("Lasso_grid", best_lasso, X_test_df, y_test)
        for alpha in [0.001, 0.01, 0.1, 1, 10]:
            candidate = make_pipeline(df, target, get_models("regression")["Lasso"].set_params(alpha=alpha))
            candidate.fit(X_train_df, y_train)
            alpha_records.append(
                {
                    "model": "Lasso",
                    "alpha": alpha,
                    "rmse": evaluate_model(f"Lasso_alpha_{alpha}", candidate, X_test_df, y_test)["rmse"],
                }
            )

        knn_pipeline = make_pipeline(df, target, models["KNN"])
        knn_grid = {"model__n_neighbors": list(range(3, 21, 2))}
        best_knn = grid_search(knn_pipeline, knn_grid, X_train_df, y_train)
        knn_metrics = evaluate_model("KNN_grid", best_knn, X_test_df, y_test)

        alpha_df = pd.DataFrame(alpha_records)
        alpha_path = os.path.join(ARTIFACTS_DIR, "alpha_analysis.csv")
        alpha_df.to_csv(alpha_path, index=False)
        mlflow.log_artifact(alpha_path)

        ridge_alpha_plot = os.path.join(REPORTS_DIR, "ridge_alpha_analysis.png")
        plot_alpha_analysis(alpha_df[alpha_df["model"] == "Ridge"], "Ridge", "rmse", ridge_alpha_plot)
        mlflow.log_artifact(ridge_alpha_plot)

        lasso_alpha_plot = os.path.join(REPORTS_DIR, "lasso_alpha_analysis.png")
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

    with mlflow.start_run(run_name="optuna_search"):
        best_rf = optuna_study(make_pipeline(df, target, models["RandomForest"]), X_train_df, y_train, n_trials=30)
        evaluate_model("RF_optuna", best_rf, X_test_df, y_test)

        best_svr = optuna_study(make_pipeline(df, target, SVR(kernel="rbf")), X_train_df, y_train, n_trials=30)
        svr_opt_metrics = evaluate_model("SVR_optuna", best_svr, X_test_df, y_test)

        best_mlp = optuna_study(make_pipeline(df, target, models["MLP_multi"]), X_train_df, y_train, n_trials=30)
        evaluate_model("MLP_optuna", best_mlp, X_test_df, y_test)

        svr_summary_path = os.path.join(ARTIFACTS_DIR, "svr_optuna_summary.json")
        save_json(
            svr_summary_path,
            {
                "best_params": json_safe(best_svr.get_params(deep=False)),
                "metrics": json_safe(svr_opt_metrics),
            },
        )
        mlflow.log_artifact(svr_summary_path)

    X_train_red, svd = apply_pca(X_train, variance=0.95)
    X_test_red = svd.transform(X_test)

    with mlflow.start_run(run_name="pca_models"):
        pca_results = []
        for name, model in models.items():
            fitted, fit_time = train_and_log(name, model, X_train_red, y_train)
            metrics = evaluate_model(name, fitted, X_test_red, y_test)
            pca_results.append({"model": name, "fit_time": fit_time, **metrics})

        pca_df = pd.DataFrame(pca_results).sort_values("rmse")
        pca_path = os.path.join(ARTIFACTS_DIR, "pca_results.json")
        pca_df.to_json(pca_path, orient="records")
        mlflow.log_artifact(pca_path)

        pca_rmse_plot_path = os.path.join(REPORTS_DIR, "pca_rmse.png")
        plot_regression_metrics(pca_df, "rmse", "RMSE after SVD dimensionality reduction", pca_rmse_plot_path)
        mlflow.log_artifact(pca_rmse_plot_path)

        pca_fit_time_plot_path = os.path.join(REPORTS_DIR, "pca_fit_times.png")
        plot_fit_times(pca_df, pca_fit_time_plot_path)
        mlflow.log_artifact(pca_fit_time_plot_path)

        fit_time_comparison_path = os.path.join(REPORTS_DIR, "fit_time_comparison_baseline_vs_svd.png")
        plot_fit_time_comparison(baseline_df, pca_df, fit_time_comparison_path)
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

    df_class = df.copy()
    df_class["high_score"] = (df_class["score"] >= 8).astype(int)
    class_target = "high_score"
    X_class_df = df_class.drop(columns=["score", class_target]).copy()
    y_class = df_class[class_target].copy()

    Xc_train_df, Xc_test_df, yc_train, yc_test = train_test_split(
        X_class_df,
        y_class,
        test_size=0.2,
        random_state=42,
    )

    preprocess_class = build_preprocess(df_class.drop(columns=["score"]), class_target)
    Xc_train = preprocess_class.fit_transform(Xc_train_df)
    Xc_test = preprocess_class.transform(Xc_test_df)
    class_feature_names = preprocess_class.get_feature_names_out().tolist()

    clf_models = get_models("classification")
    with mlflow.start_run(run_name="classification_models"):
        knn_clf_grid = {"n_neighbors": list(range(3, 22, 2)), "weights": ["uniform", "distance"]}
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

            for metric_name, metric_value in metrics.items():
                if metric_name != "model":
                    mlflow.log_metric(f"{name}_{metric_name}", metric_value)

            mlflow.sklearn.log_model(model, f"{name}_model")

            cm_path = os.path.join(REPORTS_DIR, f"{name.lower()}_confusion_matrix.png")
            plot_confusion_matrix(yc_test, preds, f"Confusion Matrix: {name}", cm_path)
            mlflow.log_artifact(cm_path)

            if name == "DecisionTreeClassifier":
                gini_tree_path = os.path.join(REPORTS_DIR, "decision_tree_gini.png")
                export_gini_tree(model, class_feature_names, gini_tree_path)
                mlflow.log_artifact(gini_tree_path)

                gini_summary_path = os.path.join(ARTIFACTS_DIR, "decision_tree_gini_summary.json")
                save_json(
                    gini_summary_path,
                    {
                        "criterion": model.criterion,
                        "tree_depth": int(model.get_depth()),
                        "n_leaves": int(model.get_n_leaves()),
                        "root_gini": float(model.tree_.impurity[0]),
                    },
                )
                mlflow.log_artifact(gini_summary_path)

        class_df = pd.DataFrame(class_results).sort_values("f1", ascending=False)
        class_path = os.path.join(ARTIFACTS_DIR, "classification_results.json")
        class_df.to_json(class_path, orient="records")
        mlflow.log_artifact(class_path)

        class_plot_path = os.path.join(REPORTS_DIR, "classification_metrics.png")
        plot_classification_metrics(class_df, class_plot_path)
        mlflow.log_artifact(class_plot_path)

        class_fit_time_plot_path = os.path.join(REPORTS_DIR, "classification_fit_times.png")
        plot_fit_times(class_df, class_fit_time_plot_path)
        mlflow.log_artifact(class_fit_time_plot_path)
        mlflow.set_tag("best_model_classification", class_df.iloc[0]["model"])

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
            "5-fold cross-validation F1 (classification)",
            cv_class_plot_path,
        )
        mlflow.log_artifact(cv_class_plot_path)

    cluster_input = X_train_red[:10000]
    with mlflow.start_run(run_name="clustering_analysis"):
        elbow_plot_path = os.path.join(REPORTS_DIR, "clustering_elbow_silhouette.png")
        cluster_results, best_k = optimal_k_elbow(cluster_input, max_k=10, output_path=elbow_plot_path)
        clustering_metrics_path = os.path.join(ARTIFACTS_DIR, "clustering_k_analysis.csv")
        cluster_results.to_csv(clustering_metrics_path, index=False)
        mlflow.log_artifact(clustering_metrics_path)
        mlflow.log_artifact(elbow_plot_path)

        labels_km, inertia = run_kmeans(cluster_input, n_clusters=best_k)
        labels_hc = run_hierarchical(cluster_input, n_clusters=best_k, linkage_method="ward")
        dendrogram_path = os.path.join(REPORTS_DIR, "hierarchical_dendrogram.png")
        plot_dendrogram(cluster_input, method="ward", sample_size=1000, output_path=dendrogram_path)
        mlflow.log_artifact(dendrogram_path)

        cluster_assignments = pd.DataFrame({"km_cluster": labels_km, "hc_cluster": labels_hc})
        cluster_assignments_path = os.path.join(ARTIFACTS_DIR, "cluster_assignments.csv")
        cluster_assignments.to_csv(cluster_assignments_path, index=False)
        mlflow.log_artifact(cluster_assignments_path)

        cluster_summary_path = os.path.join(ARTIFACTS_DIR, "clustering_summary.json")
        save_json(cluster_summary_path, {"selected_k": int(best_k), "kmeans_inertia": float(inertia)})
        mlflow.log_artifact(cluster_summary_path)

    with mlflow.start_run(run_name="recommendation_svd"):
        recommender_df = (
            df[["username", "anime_id", "score"]]
            .dropna()
            .drop_duplicates(subset=["username", "anime_id"], keep="last")
        )
        recommender_df = recommender_df.sample(
            n=min(RECOMMENDER_SAMPLE_SIZE, len(recommender_df)),
            random_state=42,
        )
        rec_train_df, rec_test_df = train_test_split(
            recommender_df,
            test_size=0.2,
            random_state=42,
        )

        recommender_model = SimpleSVD(n_factors=20, n_epochs=10, lr=0.005, reg=0.02, random_state=42)
        start = time.time()
        recommender_model.fit(rec_train_df)
        recommender_fit_time = time.time() - start

        rec_predictions = [
            recommender_model.predict(row.username, row.anime_id).est
            for row in rec_test_df.itertuples(index=False)
        ]
        recommender_metrics = compute_metrics(rec_test_df["score"], rec_predictions)
        recommender_results = pd.DataFrame(
            {
                "username": rec_test_df["username"].to_numpy(),
                "anime_id": rec_test_df["anime_id"].to_numpy(),
                "actual_score": rec_test_df["score"].to_numpy(),
                "predicted_score": rec_predictions,
            }
        )
        rec_predictions_path = os.path.join(ARTIFACTS_DIR, "recommender_predictions.csv")
        recommender_results.to_csv(rec_predictions_path, index=False)

        example_user = rec_train_df["username"].iloc[0]
        top_recommendations = get_top_n_recommendations(recommender_model, example_user, rec_train_df, n=10)
        top_recommendations_df = pd.DataFrame(top_recommendations, columns=["anime_id", "predicted_score"])
        top_recommendations_df.insert(0, "username", example_user)
        top_recommendations_path = os.path.join(ARTIFACTS_DIR, "top_recommendations_example.csv")
        top_recommendations_df.to_csv(top_recommendations_path, index=False)

        recommender_model_path = os.path.join(ARTIFACTS_DIR, "simple_svd_recommender.pkl")
        joblib.dump(recommender_model, recommender_model_path)

        recommender_summary_path = os.path.join(ARTIFACTS_DIR, "recommender_summary.json")
        save_json(
            recommender_summary_path,
            {
                "algorithm": "SimpleSVD matrix factorization",
                "features": ["username", "anime_id"],
                "target": "score",
                "fit_time": recommender_fit_time,
                "train_rows": int(len(rec_train_df)),
                "test_rows": int(len(rec_test_df)),
                "metrics": json_safe(recommender_metrics),
                "example_user": str(example_user),
            },
        )

        mlflow.log_param("algorithm", "SimpleSVD")
        mlflow.log_param("n_factors", recommender_model.n_factors)
        mlflow.log_param("n_epochs", recommender_model.n_epochs)
        mlflow.log_param("recommender_sample_size", int(len(recommender_df)))
        mlflow.log_metric("fit_time", recommender_fit_time)
        for metric_name, metric_value in recommender_metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        mlflow.log_artifact(rec_predictions_path)
        mlflow.log_artifact(top_recommendations_path)
        mlflow.log_artifact(recommender_model_path)
        mlflow.log_artifact(recommender_summary_path)

    joblib.dump(preprocess, os.path.join(ARTIFACTS_DIR, "preprocess.pkl"))
    joblib.dump(preprocess_class, os.path.join(ARTIFACTS_DIR, "preprocess_classification.pkl"))

    if not isinstance(best_rf, Pipeline) or "preprocess" not in best_rf.named_steps:
        base_rf = best_rf.named_steps["model"] if isinstance(best_rf, Pipeline) else best_rf
        best_rf = make_pipeline(df, target, base_rf)
        best_rf.fit(X_train_df, y_train)

    joblib.dump(best_rf, os.path.join(ARTIFACTS_DIR, "best_score_model.pkl"))
    best_classifier = clf_models[class_df.iloc[0]["model"]]
    joblib.dump(best_classifier, os.path.join(ARTIFACTS_DIR, "best_classification_model.pkl"))

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

        try:
            rf_model = best_rf.named_steps["model"]
            rf_feature_names = best_rf.named_steps["preprocess"].get_feature_names_out()
            export_random_forest_tree(
                rf_model,
                os.path.join(REPORTS_DIR, "rf_tree.dot"),
                os.path.join(REPORTS_DIR, "rf_tree.png"),
                feature_names=rf_feature_names,
            )
            mlflow.log_artifact(os.path.join(REPORTS_DIR, "rf_tree.png"))
        except Exception as exc:
            error_path = os.path.join(ARTIFACTS_DIR, "rf_tree_export_error.json")
            save_json(error_path, {"error": str(exc)})
            mlflow.log_artifact(error_path)


if __name__ == "__main__":
    main()
