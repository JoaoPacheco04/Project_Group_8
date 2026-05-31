import pandas as pd
import streamlit as st
import plotly.express as px
from datetime import datetime
import numpy as np

from config import COLOR_ACCENT, COLOR_ACCENT_2, COLOR_ACCENT_3, ERA_ORDER, SCALING_COLUMNS, FEATURE_DEFINITIONS
from utils.helpers import style_figure
from ml.inference import (
    predict_score_with_confidence, classify_score_with_confidence,
    get_similar_animes, extract_feature_importance, get_anime_recommendations,
    generate_report_html
)
from data.loader import get_regression_residual_source

def render_developer_view(filtered_df: pd.DataFrame, scaled_df: pd.DataFrame, df: pd.DataFrame):
    st.title("🛠️ Technical & Statistical Analysis")
    st.markdown(
        """
        <div class="caption-card">
            <strong>Developer View:</strong> Dashboard focused on data health, variance analysis, correlation matrices, and normalization checks.
        </div>
        """,
        unsafe_allow_html=True,
    )

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


def render_categorical_tab(filtered_df: pd.DataFrame, controls: dict):
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


def render_temporal_tab(filtered_df: pd.DataFrame):
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


def render_engagement_tab(filtered_df: pd.DataFrame):
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


def render_explorer_tab(filtered_df: pd.DataFrame):
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


def render_predictions_tab(filtered_df: pd.DataFrame, df: pd.DataFrame, ml_bundle: dict):
    st.subheader("🔮 Anime Score Prediction")
    if ml_bundle["score_model"] is None:
        st.info("Run `python run_all.py` to train and export the dashboard prediction artifacts.")
    else:
        regression_features = ml_bundle["feature_columns"].get("regression_features", [])
        if not regression_features:
            st.info("Missing regression feature metadata. Run `python run_all.py` to refresh dashboard artifacts.")
            return
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

                similar = get_similar_animes(df, pred, selected_title, top_n=5)
                if not similar.empty:
                    st.subheader("🎬 Similar Animes (by predicted score)")
                    st.dataframe(similar, use_container_width=True, hide_index=True)
                else:
                    st.info("No similar animes found.")
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

        st.markdown("---")
        st.subheader("📥 Batch Predictions")
        st.info("Batch predictions not available — the regression model uses user-specific features (watch status, episodes watched) that are not available in the anime metadata dataset.")


def render_classification_tab(filtered_df: pd.DataFrame, df: pd.DataFrame, ml_bundle: dict):
    st.subheader("High Score Classification")
    if ml_bundle["classification_model"] is None or ml_bundle["classification_preprocess"] is None:
        st.info("Run `python run_all.py` or `notebooks/07_classification.ipynb` to train and export the dashboard classification artifacts.")
    else:
        classification_features = ml_bundle["feature_columns"].get("classification_features", [])
        if not classification_features:
            st.info("Missing classification feature metadata. Re-run `notebooks/07_classification.ipynb` or `python run_all.py`.")
            return
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


def render_model_results_tab(model_results: dict, ml_bundle: dict):
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


def render_full_analysis_tab(model_results: dict, ml_bundle: dict, filtered_df: pd.DataFrame):
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
        st.caption("Based on a 500-row sample for performance.")
        if ml_bundle["score_model"] is None:
            st.info("Run `python run_all.py` first.")
        else:
            reg_features = ml_bundle["feature_columns"].get("regression_features", [])
            if reg_features:
                try:
                    residual_source, source_label = get_regression_residual_source(filtered_df, reg_features)
                    if residual_source.empty:
                        st.info("Residual analysis needs `ratings.csv` or matching regression features in the active dataset.")
                    else:
                        sample = residual_source.sample(min(500, len(residual_source)), random_state=42)
                        X = sample[reg_features].copy()
                        y_actual = sample["score"].values
                        y_pred = ml_bundle["score_model"].predict(X)
                        residuals = y_actual - y_pred
                        valid = ~np.isnan(residuals)
                        r, ya, yp = residuals[valid], y_actual[valid], y_pred[valid]
                        st.caption(f"Residuals computed from: {source_label}.")
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


def render_recommendations_tab(filtered_df: pd.DataFrame, df: pd.DataFrame):
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
            st.metric("Recommendations", "5", help="Fixed at 5 for consistent results")
            rec_top_n = 5

        if st.button("💡 Generate Recommendations", type="primary"):
            with st.spinner("Searching for similar anime..."):
                recommendations_df = get_anime_recommendations(df, selected_anime_rec, top_n=rec_top_n)

                if not recommendations_df.empty:
                    st.success(f"Here are the top {len(recommendations_df)} anime similar to **{selected_anime_rec}**:")
                    st.dataframe(recommendations_df, use_container_width=True, hide_index=True)
                else:
                    st.warning("Not enough recommendations were found. Try selecting a different anime.")
    else:
        st.info("No anime is available for the current filter settings.")


def render_collaborative_tab(ml_bundle: dict, df: pd.DataFrame):
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
                                    how="left"
                                )
                                df_recs_out = df_recs_out.rename(columns={"score": "global_score"})

                            st.success(f"Top {len(recommendations)} recommendations for **{username}**:")
                            st.dataframe(df_recs_out, use_container_width=True, hide_index=True)
                        else:
                            st.warning(f"No recommendations could be generated for {username}.")
