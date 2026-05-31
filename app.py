import streamlit as st

# PAGE CONFIG MUST BE THE ABSOLUTE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="MyAnimeList Dashboard",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

from config import GLOBAL_CSS, SCALING_COLUMNS
from data.loader import load_data, build_scaled_dataframe, filter_dataframe
from ml.inference import load_ml_artifacts, load_model_results
from ui.views import (
    render_developer_view,
    render_categorical_tab,
    render_temporal_tab,
    render_engagement_tab,
    render_explorer_tab,
    render_predictions_tab,
    render_classification_tab,
    render_model_results_tab,
    render_full_analysis_tab,
    render_recommendations_tab,
    render_collaborative_tab,
)

# Apply global CSS
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

def main():
    try:
        df = load_data()
    except Exception as exc:
        st.error(f"Could not load dataset files. Please check the folder and names. Error: {exc}")
        st.stop()

    filtered_df, controls, visao_dev = filter_dataframe(df)
    scaled_df = build_scaled_dataframe(filtered_df, SCALING_COLUMNS)

    if filtered_df.empty:
        st.warning("No anime matches the current filters.")
        st.stop()

    if visao_dev:
        render_developer_view(filtered_df, scaled_df, df)
    else:
        st.title("🎌 MyAnimeList Exploration Dashboard")
        st.markdown(
            """
            <div class="caption-card">
                <strong>User View:</strong> Interactive exploration focused on trends, studios, and community engagement based on the active sidebar filters.
            </div>
            """,
            unsafe_allow_html=True,
        )

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

        with tab_cat:
            render_categorical_tab(filtered_df, controls)

        with tab_time:
            render_temporal_tab(filtered_df)

        with tab_eng:
            render_engagement_tab(filtered_df)

        with tab_exp:
            render_explorer_tab(filtered_df)

        with tab_models:
            render_model_results_tab(model_results, ml_bundle)

        with tab_full:
            render_full_analysis_tab(model_results, ml_bundle, filtered_df)

        with tab_pred:
            render_predictions_tab(filtered_df, df, ml_bundle)

        with tab_clf:
            render_classification_tab(filtered_df, df, ml_bundle)

        with tab_rec:
            render_recommendations_tab(filtered_df, df)

        with tab_collab:
            render_collaborative_tab(ml_bundle, df)

if __name__ == "__main__":
    main()
