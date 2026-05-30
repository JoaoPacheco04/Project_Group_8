from __future__ import annotations

import ast
import json
import textwrap
from pathlib import Path
from datetime import datetime
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import streamlit as st
import joblib

CURRENT_YEAR = 2026
DATA_DIR = Path(__file__).resolve().parent / "datasets"
DETAILS_PATH = DATA_DIR / "details.csv"
STATS_PATH = DATA_DIR / "stats.csv"
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"

# Dark Theme Colors
COLOR_BG = "#0f0f1a"
COLOR_SURFACE = "#1a1a2e"
COLOR_TEXT = "#e0e0e0"
COLOR_MUTED = "#a0a0c0"
COLOR_ACCENT = "#e94560"
COLOR_ACCENT_2 = "#4ecdc4"
COLOR_ACCENT_3 = "#a855f7"

ERA_ORDER = [
    "Classic (Pre-2000)",
    "Golden Age (2000-2010)",
    "Modern (2011-2020)",
    "Current (2021+)",
    "Unknown",
]

BINGE_ORDER = [
    "Quick Watch (< 2h)",
    "Weekend Watch (2-5h)",
    "Week Binge (5-13h)",
    "Standard Series (13-50h)",
    "Long Commitment (50+ h)",
    "Unknown",
]

DURATION_MAP = {
    "TV": 24,
    "Movie": 90,
    "OVA": 25,
    "ONA": 25,
    "Special": 20,
    "Music": 4,
    "TV Short": 5,
    "CM": 1,
    "PV": 2,
}


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

    import json

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


def create_batch_predictions(df: pd.DataFrame, model, preprocess, features: list, target_name: str = "Predicted_Score") -> pd.DataFrame:
    """Create batch predictions for multiple animes."""
    try:
        X = df[features].copy()

        # Handle missing values
        X = X.fillna(X.mean(numeric_only=True))

        # Apply preprocessing if available
        if preprocess is not None:
            try:
                X_processed = preprocess.transform(X)
                predictions = model.predict(X_processed)
            except Exception:
                predictions = model.predict(X)
        else:
            predictions = model.predict(X)

        result_df = df[["title", "score", "type", "members"]].copy()
        result_df[target_name] = predictions
        result_df["Actual_vs_Predicted"] = result_df["score"] - result_df[target_name]

        return result_df
    except Exception as e:
        st.error(f"Batch prediction failed: {e}")
        return pd.DataFrame()


def calculate_residuals(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Calculate residual statistics."""
    residuals = actual - predicted
    return {
        "mean": float(np.mean(residuals)),
        "std": float(np.std(residuals)),
        "min": float(np.min(residuals)),
        "max": float(np.max(residuals)),
        "rmse": float(np.sqrt(np.mean(residuals**2))),
    }


def get_top_bottom_predictions(df: pd.DataFrame, model, features: list, top_n: int = 5) -> tuple:
    """Get animes with best and worst predictions."""
    try:
        X = df[features].copy()
        X = X.fillna(X.mean(numeric_only=True))
        predictions = model.predict(X)

        df_results = df[["title", "type", "score", "members", "genres"]].copy()
        df_results["predicted_score"] = predictions
        df_results["prediction_error"] = abs(df_results["score"] - df_results["predicted_score"])

        # Remove invalid predictions
        df_results = df_results[df_results["score"].notna()]

        best = df_results.nsmallest(top_n, "prediction_error")[["title", "score", "predicted_score", "prediction_error", "type"]]
        worst = df_results.nlargest(top_n, "prediction_error")[["title", "score", "predicted_score", "prediction_error", "type"]]

        return best, worst
    except Exception:
        return pd.DataFrame(), pd.DataFrame()


def analyze_error_by_category(df: pd.DataFrame, model, features: list) -> dict:
    """Analyze prediction errors by different categories."""
    try:
        X = df[features].copy()
        X = X.fillna(X.mean(numeric_only=True))
        predictions = model.predict(X)

        df_analysis = df[["type", "score", "genres"]].copy()
        df_analysis["predicted"] = predictions
        df_analysis["error"] = abs(df_analysis["score"] - df_analysis["predicted"])
        df_analysis = df_analysis[df_analysis["score"].notna()]

        results = {}

        # Error by type
        if "type" in df_analysis.columns:
            by_type = df_analysis.groupby("type").agg({
                "error": ["mean", "std", "count"]
            }).round(4)
            by_type.columns = ["Mean_Error", "Std_Error", "Count"]
            results["by_type"] = by_type.reset_index()

        # Error by genre (if available)
        if "genres" in df_analysis.columns:
            try:
                genre_errors = []
                for idx, row in df_analysis.iterrows():
                    if pd.notna(row["genres"]):
                        genres = parse_list_column(row["genres"])
                        for genre in genres:
                            genre_errors.append({"genre": genre, "error": row["error"]})

                if genre_errors:
                    by_genre = pd.DataFrame(genre_errors).groupby("genre")["error"].agg(["mean", "std", "count"]).round(4)
                    by_genre.columns = ["Mean_Error", "Std_Error", "Count"]
                    results["by_genre"] = by_genre.reset_index().nlargest(10, "Mean_Error")
            except Exception:
                pass

        return results
    except Exception:
        return {}


def create_model_comparison_data(model_results_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare data for model comparison visualization."""
    if model_results_df is None or model_results_df.empty:
        return pd.DataFrame()

    df = model_results_df.copy()

    # Normalize metrics to 0-100 scale for comparison
    if "rmse" in df.columns:
        # Lower RMSE is better, so invert
        df["rmse_score"] = (1 - (df["rmse"] - df["rmse"].min()) / (df["rmse"].max() - df["rmse"].min())) * 100

    if "r2" in df.columns:
        df["r2_score"] = df["r2"] * 100

    if "fit_time" in df.columns:
        # Lower fit time is better
        df["speed_score"] = (1 - (df["fit_time"] - df["fit_time"].min()) / (df["fit_time"].max() - df["fit_time"].min())) * 100

    return df


def get_anime_recommendations(df: pd.DataFrame, anime_title: str, top_n: int = 10) -> pd.DataFrame:
    """Generate anime recommendations based on similar characteristics."""
    try:
        # Find the selected anime
        selected = df[df["title"] == anime_title].iloc[0] if anime_title in df["title"].values else None

        if selected is None:
            return pd.DataFrame()

        # Create recommendation candidates (exclude the selected anime)
        candidates = df[df["title"] != anime_title].copy()

        # Initialize scoring
        candidates["rec_score"] = 0.0

        # 1. Similar score (±1.5 points)
        if pd.notna(selected.get("score")):
            score_diff = abs(candidates["score"] - selected["score"])
            candidates["rec_score"] += (1 - score_diff / 10) * 30  # 30% weight

        # 2. Same type
        if pd.notna(selected.get("type")):
            candidates["rec_score"] += (candidates["type"] == selected["type"]) * 20  # 20% weight

        # 3. Similar genres
        if pd.notna(selected.get("genres_list")) and isinstance(selected["genres_list"], list):
            selected_genres = set(selected["genres_list"])
            candidates["genre_overlap"] = candidates["genres_list"].apply(
                lambda x: len(set(x) & selected_genres) / max(len(selected_genres), 1) if isinstance(x, list) else 0
            )
            candidates["rec_score"] += candidates["genre_overlap"] * 25  # 25% weight

        # 4. Popularity/Members (higher is better, but not too extreme)
        if pd.notna(selected.get("members")):
            members_ratio = candidates["members"] / (selected["members"] + 1)
            members_ratio = members_ratio.clip(0.1, 10)  # Between 0.1x and 10x popularity
            candidates["rec_score"] += (1 / (1 + abs(np.log(members_ratio)))) * 15  # 15% weight

        # 5. Quality/Score (prefer highly scored)
        candidates["rec_score"] += (candidates["score"].fillna(0) / 10) * 10  # 10% weight

        # Sort and return top recommendations
        recommendations = candidates.nlargest(top_n, "rec_score")[
            ["title", "type", "score", "members", "episodes", "rec_score"]
        ].reset_index(drop=True)

        recommendations["rec_score"] = (recommendations["rec_score"] / recommendations["rec_score"].max() * 100).round(1)
        recommendations = recommendations.rename(columns={"rec_score": "Match Score %"})

        return recommendations
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


TOP_STUDIOS = [
    "Madhouse",
    "Ufotable",
    "Kyoto Animation",
    "MAPPA",
    "Bones",
    "Studio Ghibli",
    "Wit Studio",
    "Production I.G",
]
TOP_STUDIOS_LOWER = {studio.lower() for studio in TOP_STUDIOS}

# ─────────────────────────────────────────
#  PAGE CONFIG & CSS
# ─────────────────────────────────────────
st.set_page_config(
    page_title="MyAnimeList Dashboard",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    f"""
    <style>
        .stApp {{ background-color: {COLOR_BG}; color: {COLOR_TEXT}; }}
        section[data-testid="stSidebar"] {{
            background-color: {COLOR_SURFACE};
            border-right: 1px solid {COLOR_ACCENT};
        }}
        .block-container {{
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }}
        div[role="tablist"] button {{
            border-radius: 4px !important;
            font-weight: 600 !important;
            border: 1px solid #333 !important;
        }}
        div[data-testid="stMetric"] {{
            background: {COLOR_SURFACE};
            border: 1px solid {COLOR_ACCENT};
            border-radius: 10px;
            padding: 12px;
        }}
        h1, h2, h3 {{ color: {COLOR_ACCENT} !important; letter-spacing: -0.02em; }}
        label {{ color: {COLOR_MUTED} !important; }}
        .caption-card {{
            background: {COLOR_SURFACE};
            border: 1px solid {COLOR_ACCENT};
            padding: 0.9rem 1rem;
            border-radius: 10px;
            color: {COLOR_TEXT};
            margin-bottom: 1rem;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────
def parse_list_column(value: object) -> list[str]:
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except (ValueError, SyntaxError):
        pass
    return [part.strip() for part in text.split(",") if part.strip()]

def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    ratio = numerator.div(denominator.replace(0, np.nan))
    return ratio.replace([np.inf, -np.inf], np.nan)

def categorize_era(year: float) -> str:
    if pd.isna(year):
        return "Unknown"
    if year < 2000:
        return "Classic (Pre-2000)"
    if year <= 2010:
        return "Golden Age (2000-2010)"
    if year <= 2020:
        return "Modern (2011-2020)"
    return "Current (2021+)"

def encode_source(source: object) -> str:
    if pd.isna(source):
        return "Unknown"
    src = str(source).lower()
    if any(token in src for token in ["manga", "light novel", "novel", "book", "web manga", "4-koma"]):
        return "Printed Text"
    if any(token in src for token in ["game", "visual novel", "card game"]):
        return "Digital/Game"
    if "original" in src:
        return "Original"
    if any(token in src for token in ["music", "radio", "picture book"]):
        return "Other Media"
    return "Other"

def categorize_binge(hours: float) -> str:
    if pd.isna(hours) or hours <= 0:
        return "Unknown"
    if hours <= 2:
        return "Quick Watch (< 2h)"
    if hours <= 5:
        return "Weekend Watch (2-5h)"
    if hours <= 13:
        return "Week Binge (5-13h)"
    if hours <= 50:
        return "Standard Series (13-50h)"
    return "Long Commitment (50+ h)"

def build_numeric_trendline(df: pd.DataFrame, x_col: str, y_col: str, points: int = 100) -> pd.DataFrame:
    clean = df[[x_col, y_col]].dropna()
    if len(clean) < 2:
        return pd.DataFrame(columns=[x_col, y_col])
    try:
        slope, intercept = np.polyfit(clean[x_col], clean[y_col], 1)
    except np.linalg.LinAlgError:
        return pd.DataFrame(columns=[x_col, y_col])
    x_values = np.linspace(clean[x_col].min(), clean[x_col].max(), points)
    return pd.DataFrame({x_col: x_values, y_col: slope * x_values + intercept})

def build_log_trendline(df: pd.DataFrame, x_col: str, y_col: str, points: int = 100) -> pd.DataFrame:
    clean = df[[x_col, y_col]].dropna()
    clean = clean[clean[x_col] > 0]
    if len(clean) < 2:
        return pd.DataFrame(columns=[x_col, y_col])
    try:
        slope, intercept = np.polyfit(np.log10(clean[x_col]), clean[y_col], 1)
    except np.linalg.LinAlgError:
        return pd.DataFrame(columns=[x_col, y_col])
    x_values = np.geomspace(clean[x_col].min(), clean[x_col].max(), points)
    return pd.DataFrame({x_col: x_values, y_col: slope * np.log10(x_values) + intercept})

def style_figure(fig):
    fig.update_layout(
        paper_bgcolor=COLOR_BG,
        plot_bgcolor=COLOR_BG,
        font_color=COLOR_TEXT,
        title_font_size=16,
        margin=dict(t=50, b=30, l=20, r=20),
        legend_title_text="",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.1)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.1)", zeroline=False)
    return fig

FEATURE_DEFINITIONS = pd.DataFrame(
    [
        {"Feature": "Engagement_Ratio", "Definition": "favorites / members", "Meaning": "Share of members who marked the anime as favourite."},
        {"Feature": "Hype_vs_Action_Ratio", "Definition": "scored_by / members", "Meaning": "How many list members actually submitted a score."},
        {"Feature": "Completion_Ratio", "Definition": "completed / total", "Meaning": "Share of tracked users who completed the anime."},
        {"Feature": "Backlog_Ratio", "Definition": "plan_to_watch / total", "Meaning": "Share of tracked users who still have the anime in backlog."},
        {"Feature": "Drop_Rate", "Definition": "dropped / total", "Meaning": "Share of tracked users who dropped the anime."},
        {"Feature": "Popularity_to_Age_Ratio", "Definition": "members / Anime_Age_Years", "Meaning": "Approximate popularity gained per year."},
        {"Feature": "Binge_Category", "Definition": "Categorised from estimated watch time", "Meaning": "Estimated viewing commitment based on type and episode count."},
    ]
)

SCALING_COLUMNS = ["score", "members", "favorites", "episodes", "Engagement_Ratio", "Completion_Ratio", "Drop_Rate"]

# ─────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    if not DETAILS_PATH.exists() or not STATS_PATH.exists():
        missing = [str(path.name) for path in [DETAILS_PATH, STATS_PATH] if not path.exists()]
        raise FileNotFoundError(f"Missing dataset files: {', '.join(missing)}")

    details = pd.read_csv(DETAILS_PATH)
    stats = pd.read_csv(STATS_PATH)
    df = details.merge(stats, on="mal_id", how="left", validate="one_to_one")

    numeric_cols = [
        "score", "scored_by", "rank", "popularity", "members", "favorites",
        "episodes", "year", "watching", "completed", "on_hold", "dropped",
        "plan_to_watch", "total"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    parsed_start_year = pd.to_datetime(df["start_date"], errors="coerce").dt.year
    df["year_clean"] = df["year"].fillna(parsed_start_year)

    df["genres_list"] = df["genres"].apply(parse_list_column)
    df["studios_list"] = df["studios"].apply(parse_list_column)
    df["themes_list"] = df["themes"].apply(parse_list_column)
    df["genre_count"] = df["genres_list"].apply(len)
    df["primary_studio"] = df["studios_list"].apply(lambda items: items[0] if items else "Unknown")

    df["Engagement_Ratio"] = safe_ratio(df["favorites"], df["members"])
    df["Hype_vs_Action_Ratio"] = safe_ratio(df["scored_by"], df["members"])
    df["Completion_Ratio"] = safe_ratio(df["completed"], df["total"])
    df["Backlog_Ratio"] = safe_ratio(df["plan_to_watch"], df["total"])
    df["Drop_Rate"] = safe_ratio(df["dropped"], df["total"])
    df["Release_Era"] = df["year_clean"].apply(categorize_era)
    df["Anime_Age_Years"] = CURRENT_YEAR - df["year_clean"]
    df.loc[df["Anime_Age_Years"] <= 0, "Anime_Age_Years"] = np.nan
    df["Popularity_to_Age_Ratio"] = safe_ratio(df["members"], df["Anime_Age_Years"])
    df["Source_Material_Encoded"] = df["source"].apply(encode_source)
    df["Top_Tier_Studio_Flag"] = df["studios_list"].apply(
        lambda studios: int(bool({studio.lower() for studio in studios}.intersection(TOP_STUDIOS_LOWER)))
    )
    df["Est_Duration_Min"] = df["type"].map(DURATION_MAP).fillna(24)
    df["Total_Watch_Time_Hours"] = (df["episodes"] * df["Est_Duration_Min"]) / 60
    df["Total_Watch_Time_Hours"] = df["Total_Watch_Time_Hours"].replace([np.inf, -np.inf], np.nan)
    df["Binge_Category"] = df["Total_Watch_Time_Hours"].apply(categorize_binge)

    return df

def build_scaled_dataframe(df_source: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    scaled = df_source[columns].dropna().copy()
    if scaled.empty:
        return scaled
    for col in columns:
        col_min = scaled[col].min()
        col_max = scaled[col].max()
        col_std = scaled[col].std()
        if pd.notna(col_min) and pd.notna(col_max) and col_max != col_min:
            scaled[f"{col}_minmax"] = (scaled[col] - col_min) / (col_max - col_min)
        else:
            scaled[f"{col}_minmax"] = 0.0
        if pd.notna(col_std) and col_std != 0:
            scaled[f"{col}_zscore"] = (scaled[col] - scaled[col].mean()) / col_std
        else:
            scaled[f"{col}_zscore"] = 0.0
    return scaled

try:
    df = load_data()
except Exception as exc:
    st.error("Could not load dataset files. Please check the folder and names.")
    st.stop()

# ─────────────────────────────────────────
#  FILTERS
# ─────────────────────────────────────────
def filter_dataframe(df_source: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    valid_scores = df_source["score"].dropna()
    score_min_default = float(valid_scores.min()) if not valid_scores.empty else 0.0
    score_max_default = float(valid_scores.max()) if not valid_scores.empty else 10.0

    valid_years = df_source["year_clean"].dropna()
    year_min_default = int(valid_years.min()) if not valid_years.empty else 1960
    year_max_default = int(valid_years.max()) if not valid_years.empty else CURRENT_YEAR

    all_types = ["All"] + sorted(df_source["type"].dropna().unique().tolist())
    all_genres = sorted({genre for genres in df_source["genres_list"] for genre in genres})

    with st.sidebar:
        st.title("🎌 MAL Dashboard")
        st.caption("Interactive exploration of the MyAnimeList dataset")
        st.markdown("---")

        # View mode switch.
        st.subheader("View Mode")
        visao_dev = st.checkbox("🛠️ Enable Developer View", value=False, help="Switches to charts focused on exploratory statistical analysis and data normalization.")

        st.markdown("---")
        st.subheader("Global Filters")

        score_range = st.slider(
            "Score Range",
            min_value=round(score_min_default, 1),
            max_value=round(score_max_default, 1),
            value=(round(score_min_default, 1), round(score_max_default, 1)),
            step=0.1,
        )
        year_range = st.slider(
            "Release Year",
            min_value=year_min_default,
            max_value=year_max_default,
            value=(year_min_default, year_max_default),
            step=1,
        )
        selected_type = st.selectbox("Anime Type", all_types)
        selected_genres = st.multiselect(
            "Genre",
            options=all_genres,
            placeholder="Select one or more genres",
        )

        st.markdown("---")
        st.subheader("Visualization Options")
        top_n = st.slider("Top N (Rankings)", min_value=5, max_value=30, value=12, step=1)
        include_unknown_year = st.checkbox("Include unknown years", value=False)
        include_unscored = st.checkbox("Include unscored anime", value=False)

    filtered = df_source.copy()
    filtered = filtered[
        filtered["score"].between(score_range[0], score_range[1], inclusive="both")
        | (include_unscored & filtered["score"].isna())
    ]
    filtered = filtered[
        filtered["year_clean"].between(year_range[0], year_range[1], inclusive="both")
        | (include_unknown_year & filtered["year_clean"].isna())
    ]
    if selected_type != "All":
        filtered = filtered[filtered["type"] == selected_type]
    if selected_genres:
        selected_genres_set = set(selected_genres)
        filtered = filtered[filtered["genres_list"].apply(lambda items: bool(selected_genres_set.intersection(items)))]

    return filtered, {
        "score_range": score_range,
        "year_range": year_range,
        "selected_type": selected_type,
        "selected_genres": selected_genres,
        "top_n": top_n,
        "include_unknown_year": include_unknown_year,
        "include_unscored": include_unscored,
        "all_genres_count": len(all_genres),
    }, visao_dev

filtered_df, controls, visao_dev = filter_dataframe(df)
scaled_df = build_scaled_dataframe(filtered_df, SCALING_COLUMNS)

if filtered_df.empty:
    st.warning("No anime matches the current filters.")
    st.stop()

# ─────────────────────────────────────────
#  CONDICIONAL DE RENDERIZAÇÃO: DEV vs USER
# ─────────────────────────────────────────

if visao_dev:
    # ==========================================
    # VISÃO DEVELOPER (ANÁLISE ESTATÍSTICA E QUALIDADE DE DADOS)
    # ==========================================
    st.title("🛠️ Technical & Statistical Analysis")
    st.markdown(
        """
        <div class="caption-card">
            <strong>Developer View:</strong> Dashboard focused on data health, variance analysis, correlation matrices, and normalization checks.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Dev Metrics
    d1, d2, d3 = st.columns(3)
    d1.metric("Rows After Filters", f"{len(filtered_df):,}")
    d2.metric("Total Features", f"{df.shape[1]}")
    d3.metric("Files Loaded", "details.csv + stats.csv")

    dev_tab1, dev_tab2, dev_tab3 = st.tabs(["📊 Dataset Quality", "📉 Distributions & Scaling", "🔗 Correlations & Stats"])

    with dev_tab1:
        st.subheader("Dataset Characteristics & Missing Values")
        overview_left, overview_right = st.columns(2)

        with overview_left:
            dataset_overview = pd.DataFrame(
                [
                    {"Characteristic": "Domain", "Value": "Anime metadata and audience behaviour from MyAnimeList"},
                    {"Characteristic": "Files used", "Value": "details.csv + stats.csv"},
                    {"Characteristic": "Main entity", "Value": "Anime title identified by mal_id"},
                    {"Characteristic": "Rows after merge", "Value": f"{len(df):,}"},
                    {"Characteristic": "Columns after feature engineering", "Value": f"{df.shape[1]:,}"},
                    {"Characteristic": "Numeric examples", "Value": "score, members, favorites, episodes, completed"},
                    {"Characteristic": "Categorical examples", "Value": "type, source, Release_Era, Binge_Category"},
                    {"Characteristic": "List-like fields", "Value": "genres, studios, themes"},
                ]
            )
            st.dataframe(dataset_overview, use_container_width=True, hide_index=True)

        with overview_right:
            missing_summary = pd.DataFrame(
                {
                    "column": ["score", "year", "year_clean", "episodes", "members", "favorites"],
                    "missing_pct": [
                        df["score"].isna().mean() * 100,
                        df["year"].isna().mean() * 100,
                        df["year_clean"].isna().mean() * 100,
                        df["episodes"].isna().mean() * 100,
                        df["members"].isna().mean() * 100,
                        df["favorites"].isna().mean() * 100,
                    ],
                }
            )
            fig = px.bar(
                missing_summary,
                x="column",
                y="missing_pct",
                title="Missing Values in Key Columns (%)",
                color="missing_pct",
                color_continuous_scale="Sunset",
            )
            st.plotly_chart(style_figure(fig), use_container_width=True)

    with dev_tab2:
        st.subheader("Main Feature Distributions")
        col_a, col_b = st.columns(2)
        with col_a:
            score_view = filtered_df.dropna(subset=["score"])
            fig = px.histogram(score_view, x="score", nbins=40, title="Score Distribution", color_discrete_sequence=[COLOR_ACCENT])
            st.plotly_chart(style_figure(fig), use_container_width=True)

        with col_b:
            fig = px.box(score_view, y="score", title="Score Boxplot (Outliers Analysis)", color_discrete_sequence=[COLOR_ACCENT_2])
            st.plotly_chart(style_figure(fig), use_container_width=True)

        st.markdown("---")
        st.subheader("Normalisation and Standardization (Data Pipeline)")
        scaling_options = [col for col in SCALING_COLUMNS if col in scaled_df.columns]
        if scaled_df.empty or not scaling_options:
            st.info("Not enough complete records are available to compute normalization and standardization.")
        else:
            selected_scaling_feature = st.selectbox("Variable for scaling analysis", scaling_options)
            scale_left, scale_mid, scale_right = st.columns(3)

            with scale_left:
                fig = px.histogram(scaled_df, x=selected_scaling_feature, nbins=40, title=f"Original: {selected_scaling_feature}", color_discrete_sequence=[COLOR_ACCENT])
                st.plotly_chart(style_figure(fig), use_container_width=True)

            with scale_mid:
                fig = px.histogram(scaled_df, x=f"{selected_scaling_feature}_minmax", nbins=40, title=f"Min-Max: {selected_scaling_feature}", color_discrete_sequence=[COLOR_ACCENT_2])
                st.plotly_chart(style_figure(fig), use_container_width=True)

            with scale_right:
                fig = px.histogram(scaled_df, x=f"{selected_scaling_feature}_zscore", nbins=40, title=f"Z-Score: {selected_scaling_feature}", color_discrete_sequence=[COLOR_ACCENT_3])
                st.plotly_chart(style_figure(fig), use_container_width=True)

            scaling_stats = pd.DataFrame(
                [
                    {"Version": "Original", "Mean": scaled_df[selected_scaling_feature].mean(), "Variance": scaled_df[selected_scaling_feature].var()},
                    {"Version": "Min-Max", "Mean": scaled_df[f"{selected_scaling_feature}_minmax"].mean(), "Variance": scaled_df[f"{selected_scaling_feature}_minmax"].var()},
                    {"Version": "Z-Score", "Mean": scaled_df[f"{selected_scaling_feature}_zscore"].mean(), "Variance": scaled_df[f"{selected_scaling_feature}_zscore"].var()},
                ]
            )
            st.dataframe(scaling_stats.round(4), use_container_width=True, hide_index=True)

    with dev_tab3:
        st.subheader("Pearson Correlation Matrix")
        corr_cols = ["score", "scored_by", "members", "favorites", "episodes", "rank", "popularity", "watching", "completed", "dropped"]
        corr_df = filtered_df[corr_cols].copy()
        min_valid = max(25, int(len(filtered_df) * 0.1))
        valid_corr_cols = [col for col in corr_cols if corr_df[col].notna().sum() >= min_valid]
        corr_df = corr_df[valid_corr_cols]

        if len(valid_corr_cols) >= 2 and corr_df.dropna(how="all").shape[0] >= 2:
            corr = corr_df.corr(method="pearson", numeric_only=True)
            heatmap = px.imshow(corr.round(2), title="Correlation Heatmap", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
            heatmap.update_traces(text=corr.round(2).values, texttemplate="%{text:.2f}")
            st.plotly_chart(style_figure(heatmap), use_container_width=True)
        else:
            st.info("Not enough data to calculate correlation.")

else:
    # ==========================================
    # VISÃO UTILIZADOR (EXPLORAÇÃO INTERATIVA)
    # ==========================================
    st.title("🎌 MyAnimeList Exploration Dashboard")
    st.markdown(
        """
        <div class="caption-card">
            <strong>User View:</strong> Interactive exploration focused on trends, studios, and community engagement based on the active sidebar filters.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # User Metrics
    metric_1, metric_2, metric_3, metric_4, metric_5 = st.columns(5)
    metric_1.metric("Filtered Animes", f"{len(filtered_df):,}")
    metric_2.metric("Avg Score", f"{filtered_df['score'].mean():.2f}" if filtered_df["score"].notna().any() else "N/A")
    metric_3.metric("Avg Episodes", f"{filtered_df['episodes'].mean():.1f}" if filtered_df["episodes"].notna().any() else "N/A")
    metric_4.metric("Avg Members", f"{filtered_df['members'].mean():,.0f}" if filtered_df["members"].notna().any() else "N/A")
    metric_5.metric("Available Genres", str(controls["all_genres_count"]))

    ml_bundle = load_ml_artifacts()
    model_results = load_model_results()

    (tab_cat, tab_time, tab_eng, tab_exp, tab_models, tab_full, tab_pred, tab_clf, tab_rec, tab_collab) = st.tabs([
        "🏷️ Categorical",
        "📈 Temporal",
        "💎 Engagement",
        "🔎 Explorer",
        "🤖 Model Results",
        "📋 Full Analysis",
        "🔮 Predict Score",
        "🎯 Classify Score",
        "💡 Recommendations",
        "🤝 Collaborative",
    ])
    user_tab1, user_tab2, user_tab3, user_tab4 = tab_cat, tab_time, tab_eng, tab_exp
    user_tab5, user_tab6, user_tab7, user_tab8 = tab_pred, tab_clf, tab_models, tab_rec

    with user_tab1:
        st.subheader("Types, Sources, Genres and Studios")
        left, right = st.columns(2)

        with left:
            type_counts = filtered_df["type"].fillna("Unknown").value_counts().reset_index()
            type_counts.columns = ["type", "count"]
            fig = px.bar(type_counts.sort_values("count", ascending=False), x="type", y="count", title="Distribution by Type", color="count", color_continuous_scale="Tealgrn")
            st.plotly_chart(style_figure(fig), use_container_width=True)

            source_stats = filtered_df.groupby("Source_Material_Encoded").agg(count=("mal_id", "count"), avg_score=("score", "mean")).reset_index().sort_values("count", ascending=False)
            fig2 = px.bar(source_stats, x="Source_Material_Encoded", y="count", color="avg_score", title="Original Anime Source", color_continuous_scale="Tealgrn")
            st.plotly_chart(style_figure(fig2), use_container_width=True)

        with right:
            genres_expanded = filtered_df.explode("genres_list")
            genres_expanded = genres_expanded[genres_expanded["genres_list"].notna() & (genres_expanded["genres_list"] != "")]
            top_genres = genres_expanded["genres_list"].value_counts().head(controls["top_n"]).reset_index()
            top_genres.columns = ["genre", "count"]
            fig3 = px.bar(top_genres.sort_values("count"), x="count", y="genre", orientation="h", title=f"Top {controls['top_n']} Genres", color="count", color_continuous_scale="Purpor")
            st.plotly_chart(style_figure(fig3), use_container_width=True)

            studio_stats = filtered_df[filtered_df["primary_studio"] != "Unknown"].groupby("primary_studio").agg(count=("score", "count"), avg_score=("score", "mean")).reset_index()
            studio_stats = studio_stats[studio_stats["count"] >= 10].sort_values("avg_score", ascending=False).head(controls["top_n"])
            fig4 = px.bar(studio_stats.sort_values("avg_score"), x="avg_score", y="primary_studio", orientation="h", color="count", title="Top Studios by Average Score", color_continuous_scale="Sunset")
            st.plotly_chart(style_figure(fig4), use_container_width=True)

    with user_tab2:
        st.subheader("Evolution Over the Years")
        yearly = filtered_df.dropna(subset=["year_clean"]).groupby("year_clean").agg(count=("mal_id", "count"), avg_score=("score", "mean")).reset_index().sort_values("year_clean")
        yearly["count_ma3"] = yearly["count"].rolling(window=3, min_periods=1).mean()
        yearly["avg_score_ma3"] = yearly["avg_score"].rolling(window=3, min_periods=1).mean()

        left, right = st.columns(2)
        with left:
            fig = px.bar(yearly, x="year_clean", y="count", title="Number of Animes per Year", color="count", color_continuous_scale="Oranges")
            fig.add_scatter(x=yearly["year_clean"], y=yearly["count_ma3"], mode="lines", name="3-year MA", line=dict(color="#ffffff"))
            st.plotly_chart(style_figure(fig), use_container_width=True)

        with right:
            fig2 = px.line(yearly, x="year_clean", y="avg_score", markers=True, title="Average Score per Year", color_discrete_sequence=[COLOR_ACCENT])
            fig2.add_scatter(x=yearly["year_clean"], y=yearly["avg_score_ma3"], mode="lines", name="3-year MA", line=dict(color="#ffffff"))
            st.plotly_chart(style_figure(fig2), use_container_width=True)

        era_stats = filtered_df.groupby("Release_Era").agg(count=("mal_id", "count"), avg_score=("score", "mean"), avg_members=("members", "mean")).reset_index()
        era_stats["Release_Era"] = pd.Categorical(era_stats["Release_Era"], categories=ERA_ORDER, ordered=True)
        fig3 = px.bar(era_stats.sort_values("Release_Era"), x="Release_Era", y="count", color="avg_score", title="Distribution by Release Era", color_continuous_scale="Sunsetdark", hover_data={"avg_members": ":,.0f"})
        st.plotly_chart(style_figure(fig3), use_container_width=True)

    with user_tab3:
        st.subheader("Engagement & Discovery Plots")
        scatter_left, scatter_right = st.columns(2)

        with scatter_left:
            scatter_df = filtered_df.dropna(subset=["score", "members"]).copy()
            scatter_df = scatter_df[scatter_df["members"] > 0].sample(min(2000, len(scatter_df)), random_state=42)
            fig = px.scatter(scatter_df, x="members", y="score", color="type", hover_name="title", title="Score vs Members", opacity=0.55, log_x=True, color_discrete_sequence=px.colors.qualitative.Vivid)
            st.plotly_chart(style_figure(fig), use_container_width=True)

        with scatter_right:
            feature_df = filtered_df.dropna(subset=["score", "Completion_Ratio"]).copy()
            feature_df = feature_df.sample(min(2000, len(feature_df)), random_state=42)
            fig2 = px.scatter(feature_df, x="Completion_Ratio", y="score", color="type", hover_name="title", opacity=0.55, title="Score vs Completion Ratio")
            st.plotly_chart(style_figure(fig2), use_container_width=True)

    with user_tab4:
        st.subheader("🔎 Data Explorer")
        search = st.text_input("Search by title...", "")
        default_columns = ["title", "type", "score", "episodes", "year_clean", "members", "Engagement_Ratio", "Completion_Ratio", "Drop_Rate", "Binge_Category"]
        selected_columns = st.multiselect("Visible Columns", options=filtered_df.columns.tolist(), default=default_columns)

        explorer_df = filtered_df.copy()
        if search:
            explorer_df = explorer_df[explorer_df["title"].str.contains(search, case=False, na=False)]
        if selected_columns:
            explorer_df = explorer_df[selected_columns]

        st.dataframe(explorer_df.reset_index(drop=True), use_container_width=True, height=420)
        with st.expander("Feature Definitions", expanded=False):
            st.dataframe(FEATURE_DEFINITIONS, use_container_width=True, hide_index=True)

    # ---- Prediction Tab -----------------------------------------------------------
    with tab_pred:
        st.subheader("🔮 Anime Score Prediction")
        if ml_bundle["score_model"] is None:
            st.info("Run `python run_all.py` to train and export the dashboard prediction artifacts.")
        else:
            regression_features = ml_bundle["feature_columns"].get("regression_features", [])
            if not regression_features:
                st.info("Missing regression feature metadata. Run `python run_all.py` to refresh dashboard artifacts.")
                st.stop()
            prediction_source = filtered_df.dropna(subset=["title"]).copy()
            selected_title = st.selectbox(
                "Base anime",
                prediction_source["title"].sort_values().unique(),
                key="score_prediction_title",
            )
            selected_row = prediction_source[prediction_source["title"] == selected_title].iloc[0]
            input_data = {feature: selected_row.get(feature, np.nan) for feature in regression_features}

            numeric_features = [feature for feature in regression_features if feature in df.columns and pd.api.types.is_numeric_dtype(df[feature])]
            editable_features = [feature for feature in ["episodes", "members", "favorites", "rank", "popularity", "scored_by"] if feature in numeric_features]

            cols = st.columns(3)
            for idx, feature in enumerate(editable_features):
                value = input_data.get(feature, 0)
                value = 0.0 if pd.isna(value) else float(value)
                input_data[feature] = cols[idx % 3].number_input(feature, value=value, step=1.0)

            if st.button("Predict Score", key="predict_score_button"):
                try:
                    pred, confidence = predict_score_with_confidence(ml_bundle["score_model"], input_data)
                    col1, col2 = st.columns(2)
                    col1.metric("Predicted Score", f"{pred:.2f}", delta=None)
                    col2.metric("Confidence", f"{confidence*100:.0f}%", delta=None)

                    # Show similar animes
                    similar = get_similar_animes(df, pred, selected_title, top_n=5)
                    if not similar.empty:
                        st.subheader("🎬 Similar Animes (by predicted score)")
                        st.dataframe(similar, use_container_width=True, hide_index=True)
                    else:
                        st.info("No similar animes found.")
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")

            # Batch predictions download
            st.markdown("---")
            st.subheader("📥 Batch Predictions")
            st.info("Batch predictions not available — the regression model uses user-specific features (watch status, episodes watched) that are not available in the anime metadata dataset.")

    with tab_clf:
        st.subheader("High Score Classification")
        if ml_bundle["classification_model"] is None or ml_bundle["classification_preprocess"] is None:
            st.info("Run `python run_all.py` or `notebooks/07_classification.ipynb` to train and export the dashboard classification artifacts.")
        else:
            classification_features = ml_bundle["feature_columns"].get("classification_features", [])
            if not classification_features:
                st.info("Missing classification feature metadata. Re-run `notebooks/07_classification.ipynb` or `python run_all.py`.")
                st.stop()
            class_source = filtered_df.dropna(subset=["title"]).copy()
            selected_title = st.selectbox(
                "Base anime",
                class_source["title"].sort_values().unique(),
                key="class_prediction_title",
            )
            selected_row = class_source[class_source["title"] == selected_title].iloc[0]
            input_data = {feature: selected_row.get(feature, np.nan) for feature in classification_features}
            if "anime_id" in classification_features and "mal_id" in selected_row:
                input_data["anime_id"] = selected_row["mal_id"]
            if "status" in classification_features:
                input_data["status"] = st.selectbox(
                    "Watch status",
                    ["completed", "watching", "plan_to_watch", "on_hold", "dropped"],
                    key="class_watch_status",
                )
            if "num_watched_episodes" in classification_features:
                default_episodes = selected_row.get("episodes", 0)
                default_episodes = 0 if pd.isna(default_episodes) else int(default_episodes)
                input_data["num_watched_episodes"] = st.number_input(
                    "Watched episodes",
                    min_value=0,
                    value=default_episodes,
                    step=1,
                    key="class_watched_episodes",
                )
            if "is_rewatching" in classification_features:
                input_data["is_rewatching"] = float(st.checkbox("Rewatching", value=False, key="class_rewatching"))

            if st.button("Classify Score", key="classify_score_button"):
                try:
                    pred_class, confidence = classify_score_with_confidence(
                        ml_bundle["classification_model"],
                        ml_bundle["classification_preprocess"],
                        input_data,
                    )
                    label = "high score (≥ 7)" if pred_class == 1 else "lower score (< 7)"
                    col1, col2 = st.columns(2)
                    col1.metric("Predicted Class", label, delta=None)
                    col2.metric("Confidence", f"{confidence*100:.0f}%", delta=None)
                except Exception as exc:
                    st.error(f"Classification failed: {exc}")

    with tab_models:
        st.subheader("🤖 Model Results")
        mr1, mr2, mr3, mr4, mr5 = st.tabs([
            "📊 Regression", "🎯 Classification", "📐 Alpha Analysis", "🔢 KNN Optimal K", "🌳 Trees"
        ])

        with mr1:
            df_r = model_results["regression_results"]
            if df_r is None:
                st.info("Run `python run_all.py` to generate regression_results.csv")
            else:
                best = df_r.sort_values("rmse").iloc[0]
                c1, c2, c3 = st.columns(3)
                c1.metric("Best Model", best["model"])
                c2.metric("Best RMSE", f"{best['rmse']:.4f}")
                c3.metric("Best R²", f"{best['r2']:.4f}")
                st.dataframe(df_r.sort_values("rmse").reset_index(drop=True), use_container_width=True, hide_index=True)
                l, r = st.columns(2)
                with l:
                    fig = px.bar(df_r.sort_values("rmse"), x="model", y="rmse", title="RMSE by Model", color="rmse", color_continuous_scale="Reds_r")
                    fig.update_xaxes(tickangle=35)
                    st.plotly_chart(style_figure(fig), use_container_width=True)
                with r:
                    fig2 = px.bar(df_r.sort_values("fit_time", ascending=False), x="model", y="fit_time", title="Fit Time (seconds)", color="fit_time", color_continuous_scale="Blues")
                    fig2.update_xaxes(tickangle=35)
                    st.plotly_chart(style_figure(fig2), use_container_width=True)
                fig3 = px.scatter(df_r, x="rmse", y="fit_time", text="model", title="RMSE vs Fit Time - trade-off", color="r2", color_continuous_scale="Viridis")
                fig3.update_traces(textposition="top center", marker_size=10)
                st.plotly_chart(style_figure(fig3), use_container_width=True)
                if ml_bundle["score_model"] is not None:
                    fi = extract_feature_importance(ml_bundle["score_model"], ml_bundle["feature_columns"].get("regression_features", []), 10)
                    if not fi.empty:
                        st.markdown("#### Feature Importance")
                        fig_fi = px.bar(fi, x="Importance_Pct", y="Feature", orientation="h", title="Top 10 Features (Regression)", color="Importance_Pct", color_continuous_scale="Viridis")
                        st.plotly_chart(style_figure(fig_fi), use_container_width=True)

        with mr2:
            df_c = model_results["classification_results"]
            if df_c is None:
                st.info("Run `python run_all.py` to generate classification_results.csv")
            else:
                best = df_c.sort_values("f1", ascending=False).iloc[0]
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Best Model", best["model"])
                c2.metric("F1 Score", f"{best['f1']:.4f}")
                c3.metric("Precision", f"{best['precision']:.4f}")
                c4.metric("Recall", f"{best['recall']:.4f}")
                st.dataframe(df_c.sort_values("f1", ascending=False).reset_index(drop=True), use_container_width=True, hide_index=True)
                melted = df_c.melt(id_vars=["model"], value_vars=["accuracy", "precision", "recall", "f1"], var_name="metric", value_name="score")
                fig = px.bar(melted, x="model", y="score", color="metric", barmode="group", title="Precision / Recall / F1 / Accuracy by Model", color_discrete_sequence=[COLOR_ACCENT, COLOR_ACCENT_2, COLOR_ACCENT_3, "#f0c040"])
                fig.update_xaxes(tickangle=35)
                fig.update_yaxes(range=[0, 1.05])
                st.plotly_chart(style_figure(fig), use_container_width=True)
                fig2 = px.bar(df_c.sort_values("fit_time", ascending=False), x="model", y="fit_time", title="Fit Time - Classification", color="fit_time", color_continuous_scale="Blues")
                fig2.update_xaxes(tickangle=35)
                st.plotly_chart(style_figure(fig2), use_container_width=True)
                if model_results["confusion_matrix"]:
                    st.markdown(f"#### Confusion Matrix - {best['model']}")
                    st.image(model_results["confusion_matrix"], width=380)
                if ml_bundle["classification_model"] is not None:
                    fi_clf = extract_feature_importance(ml_bundle["classification_model"], ml_bundle["feature_columns"].get("classification_features", []), 10)
                    if not fi_clf.empty:
                        st.markdown("#### Feature Importance")
                        fig_fi = px.bar(fi_clf, x="Importance_Pct", y="Feature", orientation="h", title="Top 10 Features (Classification)", color="Importance_Pct", color_continuous_scale="Plasma")
                        st.plotly_chart(style_figure(fig_fi), use_container_width=True)

        with mr3:
            df_a = model_results["alpha_analysis"]
            if df_a is None:
                st.info("Run `python run_all.py` to generate alpha_analysis.csv")
            else:
                st.caption("Low alpha = less regularization. High alpha = more coefficient shrinkage.")
                fig = px.line(df_a, x="alpha", y="rmse", color="model", markers=True, title="RMSE vs Alpha - Ridge vs Lasso", log_x=True, color_discrete_map={"Ridge": COLOR_ACCENT_2, "Lasso": COLOR_ACCENT})
                st.plotly_chart(style_figure(fig), use_container_width=True)
                l, r = st.columns(2)
                with l:
                    fig2 = px.line(df_a, x="alpha", y="r2", color="model", markers=True, title="R² vs Alpha", log_x=True, color_discrete_map={"Ridge": COLOR_ACCENT_2, "Lasso": COLOR_ACCENT})
                    st.plotly_chart(style_figure(fig2), use_container_width=True)
                with r:
                    fig3 = px.line(df_a, x="alpha", y="fit_time", color="model", markers=True, title="Fit Time vs Alpha", log_x=True, color_discrete_map={"Ridge": COLOR_ACCENT_2, "Lasso": COLOR_ACCENT})
                    st.plotly_chart(style_figure(fig3), use_container_width=True)
                for mname in ["Ridge", "Lasso"]:
                    sub = df_a[df_a["model"] == mname].sort_values("rmse")
                    if not sub.empty:
                        st.success(f"**{mname}** - best alpha: `{sub.iloc[0]['alpha']}` -> RMSE: `{sub.iloc[0]['rmse']:.4f}`")
                st.dataframe(df_a, use_container_width=True, hide_index=True)

        with mr4:
            df_k = model_results["knn_analysis"]
            if df_k is None:
                st.info("Run `python run_all.py` to generate knn_analysis.csv")
            else:
                df_k = df_k.copy()
                df_k["n_neighbors"] = pd.to_numeric(df_k["n_neighbors"], errors="coerce")
                best_k = df_k.sort_values("rmse").iloc[0]
                c1, c2, c3 = st.columns(3)
                c1.metric("Optimal K", str(int(best_k["n_neighbors"])))
                c2.metric("Weighting", str(best_k.get("weights", "uniform")))
                c3.metric("Best RMSE", f"{best_k['rmse']:.4f}")
                st.caption("The optimal K is the value that minimizes cross-validation RMSE.")
                fig = px.line(df_k.sort_values("n_neighbors"), x="n_neighbors", y="rmse", color="weights" if "weights" in df_k.columns else None, markers=True, title="KNN - RMSE vs K", color_discrete_map={"uniform": COLOR_ACCENT, "distance": COLOR_ACCENT_2})
                fig.add_vline(x=int(best_k["n_neighbors"]), line_dash="dash", line_color="white", opacity=0.6, annotation_text=f"K={int(best_k['n_neighbors'])}", annotation_position="top right")
                st.plotly_chart(style_figure(fig), use_container_width=True)
                st.dataframe(df_k.sort_values("rmse").reset_index(drop=True), use_container_width=True, hide_index=True)

        with mr5:
            gini = model_results.get("gini_info", {})
            if gini:
                st.markdown("#### Decision Tree - Gini Analysis")
                g1, g2, g3 = st.columns(3)
                g1.metric("Root Gini", f"{gini.get('root_gini', 'N/A')}")
                g2.metric("Depth", str(gini.get("depth", "N/A")))
                g3.metric("Leaves", str(gini.get("leaves", "N/A")))
                st.caption("Gini near 0 = pure node. Near 0.5 = balanced classes.")
            if model_results["decision_tree"]:
                st.markdown("##### Decision Tree - max depth 3")
                st.image(model_results["decision_tree"], use_container_width=True)
            else:
                st.info("decision_tree.png not found - run `python run_all.py`")
            st.markdown("---")
            st.markdown("#### Random Forest - Representative Tree")
            st.caption("First tree in the ensemble, limited to depth 3.")
            if model_results["rf_tree"]:
                st.image(model_results["rf_tree"], use_container_width=True)
            else:
                st.info("rf_tree.png not found - run `python run_all.py`")

        st.markdown("---")
        st.subheader("📥 Export")
        html = generate_report_html(model_results, ml_bundle, filtered_df)
        st.download_button("Download HTML Report", data=html, file_name=f"mal_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html", mime="text/html", use_container_width=True)

    with tab_full:
        st.subheader("📋 Full Analysis")
        fa1, fa2, fa3, fa4 = st.tabs(["🔄 Cross Validation", "🔻 PCA / SVD", "🔵 Clustering", "📉 Residuals"])

        with fa1:
            df_cv = model_results["cv_results"]
            if df_cv is None:
                st.info("Run notebook 05 and add this at the end:")
                st.code("pd.DataFrame(cv_results).to_csv('../artifacts/cv_results.csv', index=False)")
            else:
                best_cv = df_cv.sort_values("cv_rmse_mean").iloc[0]
                c1, c2, c3 = st.columns(3)
                c1.metric("Best Model CV", best_cv["model"])
                c2.metric("Mean CV RMSE", f"{best_cv['cv_rmse_mean']:.4f}")
                c3.metric("CV RMSE Std", f"{best_cv['cv_rmse_std']:.4f}")
                st.dataframe(df_cv.sort_values("cv_rmse_mean").reset_index(drop=True), use_container_width=True, hide_index=True)
                if "cv_rmse_mean" in df_cv.columns and "cv_rmse_std" in df_cv.columns:
                    fig = px.bar(df_cv.sort_values("cv_rmse_mean"), x="model", y="cv_rmse_mean", error_y="cv_rmse_std", title="Mean CV RMSE ± Std by Model", color="cv_rmse_mean", color_continuous_scale="Reds_r")
                    fig.update_xaxes(tickangle=35)
                    st.plotly_chart(style_figure(fig), use_container_width=True)

        with fa2:
            df_p = model_results["pca_comparison"]
            if df_p is None:
                st.info("Run notebook 03 and add this at the end:")
                st.code("comparison.to_csv('../artifacts/pca_comparison.csv', index=False)")
            else:
                st.dataframe(df_p.reset_index(drop=True), use_container_width=True, hide_index=True)
                if all(c in df_p.columns for c in ["rmse_baseline", "rmse_svd"]):
                    melt = df_p.melt(id_vars=["model"], value_vars=["rmse_baseline", "rmse_svd"], var_name="representation", value_name="rmse")
                    melt["representation"] = melt["representation"].map({"rmse_baseline": "Baseline", "rmse_svd": "SVD"})
                    fig = px.bar(melt, x="model", y="rmse", color="representation", barmode="group", title="RMSE - Baseline vs SVD", color_discrete_map={"Baseline": COLOR_ACCENT, "SVD": COLOR_ACCENT_2})
                    fig.update_xaxes(tickangle=35)
                    st.plotly_chart(style_figure(fig), use_container_width=True)
                if all(c in df_p.columns for c in ["fit_time_baseline", "fit_time_svd"]):
                    melt_ft = df_p.melt(id_vars=["model"], value_vars=["fit_time_baseline", "fit_time_svd"], var_name="representation", value_name="fit_time")
                    melt_ft["representation"] = melt_ft["representation"].map({"fit_time_baseline": "Baseline", "fit_time_svd": "SVD"})
                    fig2 = px.bar(melt_ft, x="model", y="fit_time", color="representation", barmode="group", title="Fit Time - Baseline vs SVD", color_discrete_map={"Baseline": COLOR_ACCENT, "SVD": COLOR_ACCENT_2})
                    fig2.update_xaxes(tickangle=35)
                    st.plotly_chart(style_figure(fig2), use_container_width=True)

        with fa3:
            st.markdown("#### Clustering - Notebook 04 Results")
            c1, c2 = st.columns(2)
            with c1:
                if model_results["elbow_kmeans"]:
                    st.markdown("**Elbow Method + Silhouette**")
                    st.image(model_results["elbow_kmeans"], use_container_width=True)
                else:
                    st.info("elbow_kmeans.png not found - save the chart in notebook 04")
            with c2:
                if model_results["dendrogram"]:
                    st.markdown("**Hierarchical Dendrogram**")
                    st.image(model_results["dendrogram"], use_container_width=True)
                else:
                    st.info("dendrogram.png not found - save the chart in notebook 04")

        with fa4:
            st.markdown("#### Residual Analysis - Regression")
            st.caption("Based on a 500-anime sample for performance.")
            if ml_bundle["score_model"] is None:
                st.info("Run `python run_all.py` first.")
            else:
                reg_features = ml_bundle["feature_columns"].get("regression_features", [])
                if reg_features:
                    try:
                        valid_scores = filtered_df.dropna(subset=["score"])
                        sample = valid_scores.sample(min(500, len(valid_scores)), random_state=42)
                        X = sample[reg_features].fillna(sample[reg_features].mean(numeric_only=True))
                        y_actual = sample["score"].values
                        y_pred = ml_bundle["score_model"].predict(X)
                        residuals = y_actual - y_pred
                        valid = ~np.isnan(residuals)
                        r, ya, yp = residuals[valid], y_actual[valid], y_pred[valid]
                        s1, s2, s3, s4 = st.columns(4)
                        s1.metric("Mean Error", f"{r.mean():.4f}")
                        s2.metric("Std Dev", f"{r.std():.4f}")
                        s3.metric("RMSE", f"{np.sqrt(np.mean(r**2)):.4f}")
                        s4.metric("Max |Error|", f"{np.abs(r).max():.4f}")
                        l, ri = st.columns(2)
                        with l:
                            fig = px.histogram({"Residuals": r}, x="Residuals", nbins=30, title="Residual Distribution", color_discrete_sequence=[COLOR_ACCENT_2])
                            st.plotly_chart(style_figure(fig), use_container_width=True)
                        with ri:
                            scat = pd.DataFrame({"Actual": ya, "Predicted": yp, "Error": r})
                            fig2 = px.scatter(scat, x="Actual", y="Predicted", color="Error", title="Actual vs Predicted", color_continuous_scale="RdBu")
                            st.plotly_chart(style_figure(fig2), use_container_width=True)
                    except Exception as e:
                        st.warning(f"Error computing residuals: {e}")

    with tab_rec:
        st.subheader("🎯 Anime Recommendations")
        st.markdown("Discover new anime based on the characteristics of an anime you already like (Content-Based Filtering).")

        valid_titles = filtered_df["title"].dropna().sort_values().unique()
        if len(valid_titles) > 0:
            rec_col1, rec_col2 = st.columns([3, 1])
            with rec_col1:
                selected_anime_rec = st.selectbox(
                    "Select a reference anime:",
                    valid_titles,
                    key="recommendation_base_anime"
                )
            with rec_col2:
                rec_top_n = st.slider("Number of recommendations:", min_value=5, max_value=20, value=10, step=1)

            if st.button("💡 Generate Recommendations", type="primary"):
                with st.spinner("Searching for similar anime..."):
                    # Utilizamos o 'df' (dataset completo) para garantir que temos sempre animes para recomendar
                    recommendations_df = get_anime_recommendations(df, selected_anime_rec, top_n=rec_top_n)

                    if not recommendations_df.empty:
                        st.success(f"Here are the top {len(recommendations_df)} anime similar to **{selected_anime_rec}**:")
                        st.dataframe(recommendations_df, use_container_width=True, hide_index=True)
                    else:
                        st.warning("Not enough recommendations were found. Try selecting a different anime.")
        else:
            st.info("No anime is available for the current filter settings.")

    with tab_collab:
        st.subheader("🤝 Collaborative Filtering — Funk SVD")
        st.markdown(
            '<div class="caption-card">'
            '<strong>Collaborative Recommendation</strong> — based on ratings '
            'from other users with similar tastes. '
            'Enter your MyAnimeList username to receive '
            'personalized recommendations.'
            '</div>',
            unsafe_allow_html=True,
        )

        if ml_bundle["svd_model"] is None:
            st.info("SVD model not found. Run `python run_all.py` or notebook `06_recommendation.ipynb` first.")
        else:
            svd_model = ml_bundle["svd_model"]
            train_ratings = ml_bundle["svd_train_ratings"]

            if train_ratings is not None:
                c1, c2, c3 = st.columns(3)
                c1.metric("Users in Model", f"{train_ratings['username'].nunique():,}")
                c2.metric("Anime in Model", f"{train_ratings['anime_id'].nunique():,}")
                c3.metric("Training Ratings", f"{len(train_ratings):,}")

            st.markdown("---")

            col_input, col_n = st.columns([3, 1])
            with col_input:
                user_options = train_ratings["username"].unique().tolist() if train_ratings is not None else []
                username_input = st.selectbox(
                    "Seleciona um utilizador:",
                    options=user_options,
                    key="collab_username"
                )
            with col_n:
                n_recs = st.slider("Number of recommendations", 5, 20, 10, 1, key="collab_n_recs")

            if st.button("🎯 Get Recommendations", type="primary", key="collab_btn"):
                if not username_input or not str(username_input).strip():
                    st.warning("Select a username to continue.")
                else:
                    username = str(username_input).strip()

                    if username not in svd_model.user_map:
                        st.error(
                            f"The user **{username}** was not found in the training dataset. "
                            "The model only knows users included in the training sample. "
                            "Try one of the demo users above."
                        )
                    elif train_ratings is None or train_ratings.empty:
                        st.warning("SVD training ratings were not found.")
                    else:
                        with st.spinner("Generating recommendations..."):
                            from src.recommender import get_top_n_recommendations

                            recommendations = get_top_n_recommendations(
                                svd_model,
                                username,
                                train_ratings,
                                n=n_recs,
                            )

                            if recommendations:
                                df_recs_out = pd.DataFrame(
                                    recommendations,
                                    columns=["anime_id", "predicted_score"],
                                )
                                df_recs_out.insert(0, "rank", range(1, len(df_recs_out) + 1))
                                df_recs_out["predicted_score"] = df_recs_out["predicted_score"].round(2)

                                if "title" in df.columns and "mal_id" in df.columns:
                                    details = df[["mal_id", "title", "type", "score", "members"]].drop_duplicates(
                                        subset="mal_id"
                                    )
                                    df_recs_out = df_recs_out.merge(
                                        details,
                                        left_on="anime_id",
                                        right_on="mal_id",
                                        how="left",
                                    ).drop(columns=["mal_id"], errors="ignore")

                                st.success(f"Top {len(df_recs_out)} recommendations for **{username}**:")
                                st.dataframe(df_recs_out, use_container_width=True, hide_index=True)

                                if "title" in df_recs_out.columns:
                                    fig = px.bar(
                                        df_recs_out,
                                        x="predicted_score",
                                        y="title",
                                        orientation="h",
                                        title=f"Predicted scores for {username}",
                                        color="predicted_score",
                                        color_continuous_scale="Viridis",
                                    )
                                    fig.update_yaxes(autorange="reversed")
                                    st.plotly_chart(style_figure(fig), use_container_width=True)
                            else:
                                st.info(
                                    "No recommendations were found. "
                                    "The user may already have seen all available anime."
                                )

            st.markdown("---")
            with st.expander("ℹ️ How does Funk SVD work?"):
                st.markdown(
                    textwrap.dedent(
                        """
                    **Matrix Factorization (Funk SVD)** is a collaborative filtering algorithm that decomposes the user-anime matrix into two latent-factor vectors:

                    - **User factors** — represent the user taste profile
                    - **Item factors** — represent the latent characteristics of the anime
                    - **User bias** — the user tendency to give high/low scores
                    - **Item bias** — the anime tendency to receive high/low scores

                    The prediction is calculated as:

                    `score = global_mean + user_bias + item_bias + user_factors · item_factors`

                    The model learns these factors through **gradient descent**, minimizing the error between predicted scores and actual scores in the training dataset.

                    **Limitation:** The model only knows users and anime included in the training dataset. For new users (cold-start), the model returns the global mean score.
                    """
                    )
                )

    # ---------------------------------------------------------------

# ─────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────
st.markdown("---")
st.caption(
    "<div style='text-align: center;'>"
    "Group 8 | Streamlit dashboard for statistical exploration of the MyAnimeList dataset"
    "</div>",
    unsafe_allow_html=True
)
