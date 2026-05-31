import ast
import numpy as np
import pandas as pd
from config import COLOR_BG, COLOR_TEXT

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
