import numpy as np
import pandas as pd
import streamlit as st

from config import (
    DETAILS_PATH, STATS_PATH, CURRENT_YEAR, TOP_STUDIOS_LOWER, DURATION_MAP
)
from utils.helpers import (
    parse_list_column, safe_ratio, categorize_era, encode_source, categorize_binge
)

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

def filter_dataframe(df_source: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object], bool]:
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
