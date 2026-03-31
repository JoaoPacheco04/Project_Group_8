from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


CURRENT_YEAR = 2026
DATA_DIR = Path(__file__).resolve().parent / "datasets"
DETAILS_PATH = DATA_DIR / "details.csv"
STATS_PATH = DATA_DIR / "stats.csv"

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
        
        # O NOSSO NOVO CHECKBOX DE SEPARAÇÃO DA VISÃO
        st.subheader("Modo de Visualização")
        visao_dev = st.checkbox("🛠️ Ativar Visão Developer", value=False, help="Alterna para gráficos focados em análise estatística exploratória e normalização de dados.")

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
            <strong>Visão Dev:</strong> Dashboard focada na saúde dos dados, análise de variância, matrizes de correlação e validação dos modelos de normalização.
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Dev Metrics
    d1, d2, d3 = st.columns(3)
    d1.metric("Linhas (Pós-Filtros)", f"{len(filtered_df):,}")
    d2.metric("Total de Features", f"{df.shape[1]}")
    d3.metric("Ficheiros Lidos", "details.csv + stats.csv")

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
            <strong>Visão Utilizador:</strong> Exploração dinâmica interativa focada nas tendências, estúdios e engagement da comunidade baseada nos filtros ativos na barra lateral.
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

    user_tab1, user_tab2, user_tab3, user_tab4 = st.tabs(["🏷️ Categorical Insights", "📈 Temporal Trends", "💎 Engagement & Discovery", "🔎 Data Explorer"])

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