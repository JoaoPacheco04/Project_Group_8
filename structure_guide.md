# MyAnimeList Dashboard Architecture Guide

This document explains the new modular structure of the Streamlit dashboard after refactoring it from the monolithic `dashboard.py` file. The goal of this structure is to keep the codebase clean, maintainable, and production-ready by separating configuration, data loading, machine learning, and UI rendering.

## 📂 Project Structure

```text
Project_Group_8/
│
├── app.py                 # 🚀 Main entry point
├── config.py              # ⚙️ Global configuration & CSS
├── data/
│   └── loader.py          # 💾 Data ingestion & processing
├── ml/
│   └── inference.py       # 🧠 Machine learning models & prediction logic
├── ui/
│   └── views.py           # 🖥️ Streamlit UI components & tab rendering
└── utils/
    └── helpers.py         # 🛠️ Pure helper functions
```

---

## 📄 File Breakdown

### 1. `app.py`
**Purpose:** The entry point to run the Streamlit application.
- Contains the `st.set_page_config` (which must be the first Streamlit command).
- Injects the global CSS theme.
- Loads the data using functions from `data/loader.py`.
- Generates the sidebar for global filters.
- Manages routing between the "Developer View" and the standard user view.
- Sets up the tabs and calls the respective rendering functions from `ui/views.py`.
- **Run this file using:** `streamlit run app.py`

### 2. `config.py`
**Purpose:** Centralized configuration.
- Holds all mapping dictionaries (e.g., `TYPE_MAP`, `ERA_ORDER`, `TOP_STUDIOS`).
- Stores color palette constants (e.g., `COLOR_BG`, `COLOR_ACCENT`).
- Contains the `GLOBAL_CSS` string. This completely replaces the old `patch_dashboard.py` by applying the custom styling natively.
- Defines UI constants like `FEATURE_DEFINITIONS` and `SCALING_COLUMNS`.

### 3. `data/loader.py`
**Purpose:** Data ingestion, caching, and filtering.
- Contains `load_data()` which loads `details.csv` and `stats.csv` from the artifacts folder, merging them efficiently.
- Implements `@st.cache_data` heavily to ensure data is only loaded once, dramatically improving app performance.
- Contains `filter_dataframe()` which applies the user's sidebar filters (year, score, genres, type) to the main DataFrame.
- Calculates normalisation pipelines (Min-Max and Z-score).

### 4. `ml/inference.py`
**Purpose:** Machine learning predictions, loading artifacts, and modeling.
- Loads all Joblib files (`.joblib` models, scalers, imputers) using `@st.cache_resource` so ML models are loaded into memory only once per session.
- Houses functions for:
  - **Regression:** `predict_score_with_confidence()`
  - **Classification:** `classify_score_with_confidence()`
  - **Content-Based Filtering:** `get_similar_animes()` and `get_anime_recommendations()`
- Automatically handles reading and parsing `_results.csv` files generated from Jupyter notebooks.

### 5. `ui/views.py`
**Purpose:** Streamlit visual components.
- The heaviest file in terms of Streamlit logic, but highly modularized.
- Instead of one massive `if/else` block, each tab has its own dedicated function (e.g., `render_temporal_tab()`, `render_model_results_tab()`).
- Accepts the filtered dataframes as inputs and generates the Plotly charts and Streamlit widgets cleanly.

### 6. `utils/helpers.py`
**Purpose:** Decoupled helper utilities.
- Contains pure Python functions that don't depend on complex states.
- E.g., `style_figure(fig)` which applies the dashboard's consistent dark theme and layout settings to any Plotly figure.

---

## 🔄 How the Data Flows

1. **User runs `app.py`**.
2. `app.py` calls `load_data()` from `data/loader.py`.
3. The user interacts with the sidebar in `app.py`, which triggers `filter_dataframe()`.
4. `app.py` sets up the 10 UI tabs.
5. For each tab, `app.py` calls the specific `render_*` function from `ui/views.py`, passing the filtered dataframe.
6. If the user uses a prediction or recommendation tool in the UI, `ui/views.py` calls the backend logic inside `ml/inference.py`.
