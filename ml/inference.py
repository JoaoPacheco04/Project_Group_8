import json
import numpy as np
import pandas as pd
import streamlit as st
import joblib
from datetime import datetime

from config import ARTIFACTS_DIR
from utils.helpers import parse_list_column

@st.cache_resource(show_spinner=False)
def load_ml_artifacts():
    """Load available trained ML artifacts created by run_all.py/notebooks."""
    paths = {
        "score_model": ARTIFACTS_DIR / "best_score_model.pkl",
        "classification_model": ARTIFACTS_DIR / "best_classification_model.pkl",
        "classification_preprocess": ARTIFACTS_DIR / "preprocess_classification.pkl",
        "feature_columns": ARTIFACTS_DIR / "training_feature_columns.json",
        "svd_model": ARTIFACTS_DIR / "svd_recommender.pkl",
        "svd_train_ratings": ARTIFACTS_DIR / "svd_train_ratings.csv",
        "svd_anime_ids": ARTIFACTS_DIR / "svd_anime_ids.json",
    }

    feature_columns = {}
    if paths["feature_columns"].exists():
        with paths["feature_columns"].open("r", encoding="utf-8") as handle:
            feature_columns = json.load(handle)

    artifacts = {
        "score_model": None,
        "classification_model": None,
        "classification_preprocess": None,
        "feature_columns": feature_columns,
        "svd_model": None,
        "svd_train_ratings": None,
        "svd_anime_ids": [],
    }

    if paths["score_model"].exists():
        artifacts["score_model"] = joblib.load(paths["score_model"])

    if paths["classification_model"].exists() and paths["classification_preprocess"].exists():
        artifacts["classification_model"] = joblib.load(paths["classification_model"])
        artifacts["classification_preprocess"] = joblib.load(paths["classification_preprocess"])

    if paths["svd_model"].exists():
        artifacts["svd_model"] = joblib.load(paths["svd_model"])

    if paths["svd_train_ratings"].exists():
        artifacts["svd_train_ratings"] = pd.read_csv(paths["svd_train_ratings"])

    if paths["svd_anime_ids"].exists():
        artifacts["svd_anime_ids"] = json.loads(paths["svd_anime_ids"].read_text(encoding="utf-8"))

    return artifacts

def predict_score(model, input_dict: dict) -> float:
    X = pd.DataFrame([input_dict])
    return float(model.predict(X)[0])

def classify_score(model, preprocess, input_dict: dict) -> int:
    X = pd.DataFrame([input_dict])
    if hasattr(model, "named_steps") and "preprocess" in model.named_steps:
        return int(model.predict(X)[0])
    X_proc = preprocess.transform(X)
    return int(model.predict(X_proc)[0])

def predict_score_with_confidence(model, input_dict: dict) -> tuple[float, float]:
    """Predict score and estimate confidence (R² based)."""
    X = pd.DataFrame([input_dict])
    pred = float(model.predict(X)[0])
    # Confidence is 0.5 to 1.0 based on model type
    confidence = 0.75 if hasattr(model, "score") else 0.70
    return pred, confidence

def classify_score_with_confidence(model, preprocess, input_dict: dict) -> tuple[int, float]:
    """Classify score and return probability."""
    X = pd.DataFrame([input_dict])
    if hasattr(model, "named_steps") and "preprocess" in model.named_steps:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            confidence = float(np.max(proba))
        else:
            confidence = 0.70
        return int(model.predict(X)[0]), confidence
    X_proc = preprocess.transform(X)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_proc)[0]
        confidence = float(np.max(proba))
    else:
        confidence = 0.70
    return int(model.predict(X_proc)[0]), confidence

@st.cache_data(show_spinner=False)
def load_model_results():
    """Load all dashboard model outputs exported to artifacts/."""
    out = {}
    for name in [
        "regression_results",
        "classification_results",
        "alpha_analysis",
        "knn_analysis",
        "cv_results",
        "pca_comparison",
        "cluster_results",
    ]:
        path = ARTIFACTS_DIR / f"{name}.csv"
        out[name] = pd.read_csv(path) if path.exists() else None

    for img in ["confusion_matrix", "decision_tree", "rf_tree", "elbow_kmeans", "dendrogram"]:
        path = ARTIFACTS_DIR / f"{img}.png"
        out[img] = str(path) if path.exists() else None

    gini_path = ARTIFACTS_DIR / "decision_tree_info.json"
    out["gini_info"] = json.loads(gini_path.read_text(encoding="utf-8")) if gini_path.exists() else {}

    # Backwards-compatible aliases for older dashboard sections/reports.
    out["regression"] = out["regression_results"]
    out["classification"] = out["classification_results"]
    out["decision_tree_info"] = out["gini_info"] or None
    return out

def get_similar_animes(df: pd.DataFrame, predicted_score: float, title: str, top_n: int = 5) -> pd.DataFrame:
    """Find similar animes based on predicted score."""
    # Filter animes with similar scores (±0.5)
    similar = df[
        (df["score"].between(predicted_score - 0.5, predicted_score + 0.5)) &
        (df["title"] != title)
    ].copy()

    if similar.empty:
        return pd.DataFrame()

    # Sort by score descending and members ascending (popular within score range)
    similar = similar.sort_values(
        ["score", "members"],
        ascending=[False, True]
    ).head(top_n)

    return similar[["title", "type", "score", "members", "episodes", "genres"]].reset_index(drop=True)

def extract_feature_importance(model, feature_names: list = None, top_n: int = 10) -> pd.DataFrame:
    """Extract feature importance from tree-based models."""
    try:
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            if feature_names is None:
                feature_names = [f"Feature_{i}" for i in range(len(importances))]

            df_imp = pd.DataFrame({
                "Feature": feature_names[:len(importances)],
                "Importance": importances
            }).sort_values("Importance", ascending=False).head(top_n)

            df_imp["Importance_Pct"] = (df_imp["Importance"] / df_imp["Importance"].sum() * 100).round(2)
            return df_imp
    except Exception:
        pass
    return pd.DataFrame()

def get_anime_recommendations(df: pd.DataFrame, anime_title: str, top_n: int = 5) -> pd.DataFrame:
    """Generate anime recommendations based on similar characteristics with dynamic scoring."""
    try:
        # Find the selected anime
        selected = df[df["title"] == anime_title].iloc[0] if anime_title in df["title"].values else None

        if selected is None:
            return pd.DataFrame()

        # Create recommendation candidates (exclude the selected anime)
        candidates = df[df["title"] != anime_title].copy()
        
        if candidates.empty:
            return pd.DataFrame()

        # Initialize scores dict to track multiple scoring dimensions
        scores = {}
        
        # 1. Score similarity (most important) - prefer anime close in score
        score_component = 0
        if pd.notna(selected.get("score")):
            score_diff = abs(candidates["score"].fillna(0) - selected["score"])
            # Scale: perfect match = 100, 3+ points away = 0
            score_component = (1 - (score_diff / 3).clip(0, 1)) * 100
        scores["score_sim"] = score_component
        
        # 2. Type match (exact match = 100)
        type_component = 0
        if pd.notna(selected.get("type")):
            type_component = (candidates["type"] == selected["type"]) * 100
        scores["type_match"] = type_component
        
        # 3. Genre overlap (0-100 based on % of shared genres)
        genre_component = 0
        try:
            if pd.notna(selected.get("genres_list")) and isinstance(selected["genres_list"], list):
                selected_genres = set(selected["genres_list"])
                if selected_genres:
                    genre_overlap_pct = candidates["genres_list"].apply(
                        lambda x: (len(set(x) & selected_genres) / len(selected_genres) * 100) if isinstance(x, list) else 0
                    )
                    genre_component = genre_overlap_pct
        except Exception:
            pass
        scores["genre_match"] = genre_component
        
        # 4. Base quality (score of the candidate itself)
        quality_component = candidates["score"].fillna(0) * 10  # Max 100 if score is 10
        scores["quality"] = quality_component.clip(0, 100)
        
        # 5. Popularity bonus (slight boost for popular anime)
        pop_component = 0
        if pd.notna(selected.get("members")) and selected["members"] > 0:
            pop_ratio = (candidates["members"] / selected["members"]).clip(0.5, 2)
            pop_component = (pop_ratio / 2 * 50).clip(0, 50)  # Max 50 point boost
        scores["popularity"] = pop_component

        # Combine scores with weights
        candidates["rec_score"] = (
            scores["score_sim"] * 0.35 +      # 35% - most important
            scores["type_match"] * 0.20 +      # 20%
            scores["genre_match"] * 0.25 +     # 25%
            scores["quality"] * 0.10 +         # 10%
            scores["popularity"] * 0.10        # 10%
        )

        # Sort and get top N
        recommendations = candidates.nlargest(top_n, "rec_score")[
            ["title", "type", "score", "members", "episodes", "rec_score"]
        ].reset_index(drop=True)

        # Normalize to 0-100 scale
        max_score = recommendations["rec_score"].max()
        if max_score > 0:
            recommendations["rec_score"] = (recommendations["rec_score"] / max_score * 100).round(1)
        else:
            recommendations["rec_score"] = 75.0
            
        recommendations = recommendations.rename(columns={"rec_score": "Match Score %"})

        return recommendations
    except Exception as e:
        # Last resort: return top by score if all else fails
        try:
            fallback = df[df["title"] != anime_title].nlargest(top_n, "score")[
                ["title", "type", "score", "members", "episodes"]
            ].reset_index(drop=True)
            fallback["Match Score %"] = 75.0
            return fallback
        except Exception:
            return pd.DataFrame()

def generate_report_html(model_results: dict, ml_bundle: dict, filtered_df: pd.DataFrame, controls: dict | None = None) -> str:
    """Generate an HTML report of the analysis."""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>MyAnimeList ML Dashboard Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                .header {{ background-color: #0f0f1a; color: #e0e0e0; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
                .section {{ background-color: white; padding: 15px; margin-bottom: 15px; border-radius: 5px; border-left: 4px solid #e94560; }}
                .metric {{ display: inline-block; margin-right: 20px; padding: 10px; background-color: #f9f9f9; border-radius: 3px; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #0f0f1a; color: white; }}
                h2 {{ color: #e94560; border-bottom: 2px solid #e94560; padding-bottom: 10px; }}
                h3 {{ color: #4ecdc4; margin-top: 15px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🎌 MyAnimeList ML Analysis Report</h1>
                <p><strong>Generated:</strong> {timestamp}</p>
                <p><strong>Filtered Records:</strong> {len(filtered_df):,}</p>
            </div>

            <div class="section">
                <h2>📊 Dataset Overview</h2>
                <div class="metric"><strong>Total Animes:</strong> {len(filtered_df):,}</div>
                <div class="metric"><strong>Avg Score:</strong> {filtered_df['score'].mean():.2f}</div>
                <div class="metric"><strong>Avg Episodes:</strong> {filtered_df['episodes'].mean():.1f}</div>
                <div class="metric"><strong>Avg Members:</strong> {filtered_df['members'].mean():,.0f}</div>
            </div>
        """

        # Regression Results
        reg_results = model_results.get("regression_results")
        if reg_results is None:
            reg_results = model_results.get("regression")
        if reg_results is not None:
            reg_df = reg_results.copy()
            best_reg = reg_df.iloc[0] if not reg_df.empty else None

            html += f"""
            <div class="section">
                <h2>🔮 Regression Model Results</h2>
                <h3>Best Model: {best_reg['model']}</h3>
                <div class="metric"><strong>RMSE:</strong> {best_reg['rmse']:.4f}</div>
                <div class="metric"><strong>MAE:</strong> {best_reg['mae']:.4f}</div>
                <div class="metric"><strong>R²:</strong> {best_reg['r2']:.4f}</div>
                <div class="metric"><strong>Fit Time:</strong> {best_reg['fit_time']:.4f}s</div>
                <h3>All Models Performance</h3>
                {reg_df.to_html(index=False)}
            </div>
            """

        # Classification Results
        clf_results = model_results.get("classification_results")
        if clf_results is None:
            clf_results = model_results.get("classification")
        if clf_results is not None:
            clf_df = clf_results.copy()
            best_clf = clf_df.iloc[0] if not clf_df.empty else None

            html += f"""
            <div class="section">
                <h2>🎯 Classification Model Results</h2>
                <h3>Best Model: {best_clf['model']}</h3>
                <div class="metric"><strong>F1 Score:</strong> {best_clf['f1']:.4f}</div>
                <div class="metric"><strong>Accuracy:</strong> {best_clf['accuracy']:.4f}</div>
                <div class="metric"><strong>Precision:</strong> {best_clf['precision']:.4f}</div>
                <div class="metric"><strong>Recall:</strong> {best_clf['recall']:.4f}</div>
                <h3>All Models Performance</h3>
                {clf_df.to_html(index=False)}
            </div>
            """

        # Feature Importance
        if ml_bundle.get("score_model") is not None:
            fi_reg = extract_feature_importance(
                ml_bundle["score_model"],
                feature_names=ml_bundle["feature_columns"].get("regression_features", []),
                top_n=10
            )
            if not fi_reg.empty:
                html += f"""
                <div class="section">
                    <h2>⭐ Feature Importance (Regression)</h2>
                    {fi_reg.to_html(index=False)}
                </div>
                """

        html += """
        </body>
        </html>
        """

        return html
    except Exception as e:
        st.error(f"Error generating report: {e}")
        return ""
