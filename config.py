import pandas as pd
from pathlib import Path

CURRENT_YEAR = 2026
# Assuming app.py is at the root, so __file__ resolves to Project_Group_8/config.py
DATA_DIR = Path(__file__).resolve().parent / "datasets"
DETAILS_PATH = DATA_DIR / "details.csv"
STATS_PATH = DATA_DIR / "stats.csv"
RATINGS_PATH = DATA_DIR / "ratings.csv"
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

RATING_MODEL_DTYPES = {
    "username": "category",
    "anime_id": "int32",
    "status": "category",
    "score": "float32",
    "is_rewatching": "float32",
    "num_watched_episodes": "float32",
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

# Combined and Enhanced CSS from patch_dashboard.py
GLOBAL_CSS = f"""
    <style>
        /* Base */
        .stApp {{ background-color: {COLOR_BG}; color: {COLOR_TEXT}; }}
        section[data-testid="stSidebar"] {{
            background-color: {COLOR_SURFACE};
            border-right: 1px solid rgba(233,69,96,0.35);
        }}
        .block-container {{
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }}

        /* Typography */
        h1, h2, h3 {{ color: {COLOR_ACCENT} !important; letter-spacing: -0.02em; }}
        label {{ color: {COLOR_MUTED} !important; }}

        /* Tabs */
        div[role="tablist"] button {{
            border-radius: 6px !important;
            font-weight: 600 !important;
            border: 1px solid rgba(255,255,255,0.08) !important;
            transition: background 0.18s ease, border-color 0.18s ease !important;
        }}
        div[role="tablist"] button:hover {{
            background: rgba(233,69,96,0.15) !important;
            border-color: {COLOR_ACCENT} !important;
            color: #ffffff !important;
        }}
        div[role="tablist"] button[aria-selected="true"] {{
            background: rgba(233,69,96,0.22) !important;
            border-color: {COLOR_ACCENT} !important;
            color: #ffffff !important;
        }}

        /* Buttons */
        .stButton > button {{
            background: linear-gradient(135deg, {COLOR_ACCENT}, {COLOR_ACCENT_3}) !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            letter-spacing: 0.01em !important;
            transition: transform 0.15s ease, box-shadow 0.15s ease !important;
            box-shadow: 0 4px 14px rgba(233,69,96,0.35) !important;
        }}
        .stButton > button:hover {{
            transform: translateY(-2px) !important;
            box-shadow: 0 7px 22px rgba(233,69,96,0.5) !important;
        }}
        .stButton > button:active {{
            transform: translateY(0px) !important;
        }}

        /* Metric cards */
        div[data-testid="stMetric"] {{
            background: linear-gradient(135deg, {COLOR_SURFACE}, #16213e);
            border: 1px solid rgba(233,69,96,0.35);
            border-radius: 10px;
            padding: 12px;
            box-shadow: 0 4px 14px rgba(233,69,96,0.08),
                        inset 0 1px 0 rgba(255,255,255,0.04);
            transition: border-color 0.18s ease, box-shadow 0.18s ease;
        }}
        div[data-testid="stMetric"]:hover {{
            border-color: rgba(233,69,96,0.65) !important;
            box-shadow: 0 6px 20px rgba(233,69,96,0.18),
                        0 0 0 1px rgba(233,69,96,0.1) !important;
        }}

        /* Input fields */
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input {{
            background-color: {COLOR_SURFACE} !important;
            border: 1px solid rgba(233,69,96,0.3) !important;
            color: {COLOR_TEXT} !important;
            border-radius: 6px !important;
            transition: border-color 0.15s ease !important;
        }}
        .stTextInput > div > div > input:focus,
        .stNumberInput > div > div > input:focus {{
            border-color: {COLOR_ACCENT} !important;
            box-shadow: 0 0 0 2px rgba(233,69,96,0.2) !important;
        }}

        /* Dataframe */
        div[data-testid="stDataFrame"] {{
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid rgba(255,255,255,0.07);
        }}

        /* Caption card */
        .caption-card {{
            background: {COLOR_SURFACE};
            border: 1px solid rgba(233,69,96,0.35);
            padding: 0.9rem 1rem;
            border-radius: 10px;
            color: {COLOR_TEXT};
            margin-bottom: 1rem;
        }}
    </style>
"""
