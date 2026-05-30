"""
Script to apply remaining dashboard.py edits that multi_replace can't handle
due to Windows CRLF line endings.
"""
from pathlib import Path

p = Path("dashboard.py")
content = p.read_bytes().decode("utf-8")

# ── 1. Remove the confusing user_tab5–8 alias line ──────────────────────────
old = (
    "    user_tab1, user_tab2, user_tab3, user_tab4 = tab_cat, tab_time, tab_eng, tab_exp\r\n"
    "    user_tab5, user_tab6, user_tab7, user_tab8 = tab_pred, tab_clf, tab_models, tab_rec\r\n"
)
new = (
    "    user_tab1, user_tab2, user_tab3, user_tab4 = tab_cat, tab_time, tab_eng, tab_exp\r\n"
)
assert old in content, "Alias block not found!"
content = content.replace(old, new, 1)
print("[OK] Removed user_tab5-8 alias line")

# ── 2. Replace CSS block ─────────────────────────────────────────────────────
old_css_start = "st.markdown(\r\n    f\"\"\"\r\n    <style>\r\n        .stApp"
old_css_end   = "    unsafe_allow_html=True,\r\n)\r\n\r\n# ─────────────────────────────────────────\r\n#  HELPER FUNCTIONS"

assert old_css_start in content, "CSS start not found!"
assert old_css_end   in content, "CSS end not found!"

# Find the block boundaries
start_idx = content.index(old_css_start)
end_idx   = content.index(old_css_end)

new_css = '''st.markdown(
    f"""
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
    """,
    unsafe_allow_html=True,
)\r\n\r\n# -----------------------------------------\r\n#  HELPER FUNCTIONS'''

content = content[:start_idx] + new_css + content[end_idx + len(old_css_end):]
print("[OK] CSS block replaced with enhanced version")

p.write_bytes(content.encode("utf-8"))
print("[OK] dashboard.py saved")

# Verify
content_check = p.read_text(encoding="utf-8")
checks = [
    "rgba(233,69,96,0.15)",
    "linear-gradient(135deg",
    "box-shadow: 0 4px 14px",
    "user_tab5" not in content_check,  # should be gone
]
all_ok = all(checks)
print(f"Verification: {'PASS' if all_ok else 'FAIL'}")
print(f"  Tab hover CSS present: {'rgba(233,69,96,0.15)' in content_check}")
print(f"  Button gradient present: {'linear-gradient(135deg' in content_check}")
print(f"  Metric shadow present: {'box-shadow: 0 4px 14px' in content_check}")
print(f"  user_tab5 removed: {'user_tab5' not in content_check}")
