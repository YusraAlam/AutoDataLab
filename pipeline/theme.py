"""Color themes, global CSS and chart palettes for AutoDataLab."""
import streamlit as st
import plotly.express as px

THEMES = {
    "🔥 Sunset Ember": dict(
        dark=True, bg1="#1C0F0D", bg2="#3A1A14", text="#FFF7ED", muted="#FDBA74",
        accent="#FB923C", accent2="#F43F5E", card="rgba(68,32,24,0.75)", border="#7C2D12",
        sidebar="#140A08", input="#3A1D16", btn_text="#1C0F0D", scale="YlOrRd",
        colors=["#FB923C", "#F43F5E", "#FACC15", "#FB7185", "#38BDF8", "#34D399", "#C084FC", "#F97316"],
    ),
}
DEFAULT_THEME = next(iter(THEMES))

QUALITATIVE_PALETTES = {
    "Plotly": px.colors.qualitative.Plotly,
    "Vivid": px.colors.qualitative.Vivid,
    "Bold": px.colors.qualitative.Bold,
    "Safe (color-blind friendly)": px.colors.qualitative.Safe,
    "D3": px.colors.qualitative.D3,
    "Set2 (soft)": px.colors.qualitative.Set2,
    "Pastel": px.colors.qualitative.Pastel,
    "Antique": px.colors.qualitative.Antique,
    "Prism": px.colors.qualitative.Prism,
    "T10": px.colors.qualitative.T10,
}

CONTINUOUS_SCALES = [
    "Viridis", "Plasma", "Inferno", "Magma", "Cividis", "Turbo", "Blues", "Teal", "Tealgrn",
    "Sunset", "YlOrRd", "Purples", "Greens", "RdBu", "Spectral", "Portland", "ice",
]

AUTO_TEMPLATE = "Auto (match app theme)"
CHART_TEMPLATES = [AUTO_TEMPLATE, "plotly_white", "plotly_dark", "ggplot2", "seaborn",
                   "simple_white", "presentation", "plotly", "none"]


def get_theme() -> dict:
    name = st.session_state.get("theme_name", DEFAULT_THEME)
    return THEMES.get(name, THEMES[DEFAULT_THEME])


def render_theme_selector():
    """Single fixed theme (Sunset Ember) - no selector is shown."""
    return None


def chart_template() -> str:
    return "plotly_dark" if get_theme()["dark"] else "plotly_white"


def theme_colors() -> list:
    return list(get_theme()["colors"])


def color_swatches_html(colors) -> str:
    chips = "".join(
        f'<span style="display:inline-block;width:28px;height:28px;border-radius:8px;margin-right:6px;'
        f'background:{c};border:1px solid rgba(128,128,128,.4);"></span>' for c in colors
    )
    return f'<div style="margin:6px 0 10px 0;">{chips}</div>'


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


_STATIC_CSS = """
.stApp { background: linear-gradient(135deg, var(--bg1), var(--bg2)); background-attachment: fixed; color: var(--text); }
.stApp, .stApp p, .stApp li, .stApp label, .stApp [data-testid="stMarkdownContainer"] { color: var(--text); }
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 { color: var(--text); }
.stApp [data-testid="stCaptionContainer"], .stApp small { color: var(--muted) !important; }
header[data-testid="stHeader"] { background: transparent; }
hr { border-color: var(--border) !important; }

section[data-testid="stSidebar"] { background: var(--sidebar); border-right: 1px solid var(--border); }
section[data-testid="stSidebar"] * { color: var(--text); }

.main-title { font-size: 42px; font-weight: 800; text-align: center; margin-bottom: 4px;
  background: linear-gradient(90deg, var(--accent), var(--accent2));
  -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; }
.subtitle { text-align: center; font-size: 16px; color: var(--muted); margin-bottom: 24px; }

/* Buttons */
.stButton > button, .stDownloadButton > button {
  background: var(--card); color: var(--text); border: 1px solid var(--accent);
  border-radius: 10px; font-weight: 600; transition: all .2s ease; }
.stButton > button:hover, .stDownloadButton > button:hover {
  background: var(--accent); color: var(--btn-text); transform: translateY(-2px);
  box-shadow: 0 8px 20px var(--glow); border-color: var(--accent); }
.stButton > button:active, .stDownloadButton > button:active { transform: translateY(0); }
.stButton > button *, .stDownloadButton > button * { color: inherit !important; }
.stButton > button:disabled { opacity: .45; }

/* Tabs */
button[data-baseweb="tab"] { color: var(--muted); font-weight: 600; transition: color .2s ease; }
button[data-baseweb="tab"]:hover { color: var(--accent); }
button[data-baseweb="tab"][aria-selected="true"], button[data-baseweb="tab"][aria-selected="true"] * { color: var(--accent) !important; }
div[data-baseweb="tab-highlight"] { background-color: var(--accent) !important; }
div[data-baseweb="tab-border"] { background-color: var(--border) !important; }

/* Section selector (persistent tabs) */
div[data-testid="stRadio"] div[role="radiogroup"][aria-orientation="horizontal"],
div[data-testid="stRadio"] > div[role="radiogroup"] { gap: 6px; flex-wrap: wrap; }

/* Metrics */
div[data-testid="stMetric"] { background: var(--card); border: 1px solid var(--border);
  border-radius: 12px; padding: 12px 16px; transition: transform .2s ease, box-shadow .2s ease; }
div[data-testid="stMetric"]:hover { transform: translateY(-3px); box-shadow: 0 8px 22px var(--glow); }
div[data-testid="stMetricValue"], div[data-testid="stMetricValue"] * { color: var(--accent) !important; }
div[data-testid="stMetricLabel"], div[data-testid="stMetricLabel"] * { color: var(--muted) !important; }

/* Expanders & containers */
div[data-testid="stExpander"] { background: var(--card); border: 1px solid var(--border); border-radius: 12px; }
div[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 10px; overflow: hidden; }
div[data-testid="stAlert"] { border-radius: 10px; }

/* Inputs */
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div, div[data-baseweb="textarea"] > div {
  background: var(--input) !important; border-color: var(--border) !important; color: var(--text) !important; }
div[data-baseweb="select"] *, div[data-baseweb="input"] input, textarea { color: var(--text) !important; }
div[data-baseweb="select"] > div:hover, div[data-baseweb="input"] > div:hover { border-color: var(--accent) !important; }
div[data-baseweb="popover"] ul, div[data-baseweb="menu"] { background: var(--input) !important; }
div[data-baseweb="popover"] li { color: var(--text) !important; }
div[data-baseweb="popover"] li:hover { background: var(--glow) !important; }
span[data-baseweb="tag"] { background: var(--accent) !important; }
span[data-baseweb="tag"] * { color: var(--btn-text) !important; }
div[data-baseweb="slider"] div[role="slider"] { background-color: var(--accent) !important; }
div[data-testid="stFileUploaderDropzone"] { background: var(--card); border: 2px dashed var(--accent); border-radius: 12px; }
div[data-testid="stFileUploaderDropzone"] * { color: var(--text) !important; }

/* Progress */
.stProgress > div > div > div > div { background-image: linear-gradient(90deg, var(--accent), var(--accent2)); }

/* Dashboard cards */
.pbi-card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 16px;
  text-align: center; transition: transform .2s ease, box-shadow .2s ease; }
.pbi-card:hover { transform: translateY(-4px); box-shadow: 0 10px 24px var(--glow); }
.pbi-card-title { font-size: 12px; color: var(--muted); font-weight: 700; text-transform: uppercase; letter-spacing: .5px; }
.pbi-card-value { font-size: 26px; color: var(--accent); font-weight: 800; margin-top: 6px; }
.slicer-container { background: var(--card); padding: 14px; border-radius: 10px; border-left: 5px solid var(--accent); margin-bottom: 20px; }

/* Scrollbar */
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-thumb { background: var(--accent); border-radius: 8px; }
::-webkit-scrollbar-track { background: transparent; }
"""


def inject_css():
    t = get_theme()
    variables = (
        ":root {"
        f"--bg1:{t['bg1']};--bg2:{t['bg2']};--text:{t['text']};--muted:{t['muted']};"
        f"--accent:{t['accent']};--accent2:{t['accent2']};--card:{t['card']};--border:{t['border']};"
        f"--sidebar:{t['sidebar']};--input:{t['input']};--btn-text:{t['btn_text']};"
        f"--glow:{_rgba(t['accent'], 0.28)};"
        "}"
    )
    st.markdown(f"<style>{variables}{_STATIC_CSS}</style>", unsafe_allow_html=True)
