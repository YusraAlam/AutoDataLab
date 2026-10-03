import traceback

import streamlit as st

from pipeline.cleaner import DataCleanerEngine
from pipeline.data_explorer import DataExplorerEngine
from pipeline.data_loader import DataLoader
from pipeline.eda_dashboard import PowerBIDashboardEngine
from pipeline.feature_engineering import FeatureEngineeringEngine
from pipeline.ml_tuner import AutoMLEngine
from pipeline.theme import inject_css, render_theme_selector
from pipeline.utils import show_flash

try:  # chatbot is optional so a problem there never takes the whole app down
    from pipeline.chatbot import DataChatbotEngine
    CHATBOT_IMPORT_ERROR = None
except Exception as exc:  # noqa: BLE001
    DataChatbotEngine = None
    CHATBOT_IMPORT_ERROR = exc

st.set_page_config(page_title="AutoDataLab Enterprise", page_icon="⚡", layout="wide")

PAGES = [
    "📂  Data Loader",
    "🔍  Data Explorer",
    "🧹  Data Cleaner",
    "⚙️  Feature Engineering",
    "📊  Dashboard",
    "🤖  ML & Hyperparameter Tuning",
    "💬  AI Data Chatbot",
]


def render_page(app_mode, loader, df):
    if app_mode == PAGES[0]:
        loader.render_upload_ui()
    elif df is None:
        st.warning("⚠️  Please Upload Dataset First")
    elif app_mode == PAGES[1]:
        DataExplorerEngine(df).render_explorer_interface()
    elif app_mode == PAGES[2]:
        DataCleanerEngine(df).render_cleaning_interface()
    elif app_mode == PAGES[3]:
        FeatureEngineeringEngine(df).render_interface()
    elif app_mode == PAGES[4]:
        PowerBIDashboardEngine(df).render_dashboard()
    elif app_mode == PAGES[5]:
        AutoMLEngine(df).render_studio()
    elif app_mode == PAGES[6]:
        if DataChatbotEngine is None:
            st.error(f"❌ The chatbot module could not be loaded: {CHATBOT_IMPORT_ERROR}")
        else:
            DataChatbotEngine(df).render_chat_interface()


def main():
    st.session_state["_transparent_charts"] = True  # the EDA colour studio may switch this off per run

    st.sidebar.title("🛠️ AutoDataLab")
    app_mode = st.sidebar.radio("Select Step", PAGES, key="app_mode")
    render_theme_selector()
    inject_css()

    st.markdown('<div class="main-title">⚡ AutoDataLab Enterprise</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Upload → Explore → Clean → Engineer → Visualize → Model → Ask</div>',
                unsafe_allow_html=True)

    loader = DataLoader()
    df = loader.get_data()

    st.sidebar.divider()
    if df is not None:
        st.sidebar.success(f"✅ Dataset Loaded: `{st.session_state.get('file_name', 'dataset')}`")
        st.sidebar.caption(f"{df.shape[0]:,} rows × {df.shape[1]} columns")
        if "ml_best_model" in st.session_state:
            st.sidebar.info(f"🤖 Best Model Ready: `{st.session_state.get('ml_best_name', 'trained')}`")
    else:
        st.sidebar.warning("⚠️ No dataset loaded yet")

    show_flash()

    try:
        render_page(app_mode, loader, df)
    except Exception as exc:  # noqa: BLE001  (st.rerun() is not an Exception subclass)
        st.error(f"❌ Something went wrong on this page: {exc}")
        with st.expander("Technical details"):
            st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
