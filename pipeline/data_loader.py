import streamlit as st
import pandas as pd

from pipeline.utils import (
    clear_ml_state, flash, ui_button, ui_dataframe, ui_download,
)


class DataLoader:
    def __init__(self):
        defaults = {
            "df": None,
            "original_df": None,
            "file_name": None,
            "file_sig": None,
            "history": [],  # undo stack of DataFrame snapshots
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    def get_data(self) -> pd.DataFrame:
        """Return the current DataFrame."""
        return st.session_state.df

    # ------------------------------------------------------------------ reading
    @staticmethod
    def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Make column names strings, strip spaces, and de-duplicate."""
        seen = {}
        new_cols = []
        for col in df.columns:
            name = str(col).strip() or "unnamed"
            if name in seen:
                seen[name] += 1
                name = f"{name}_{seen[name]}"
            else:
                seen[name] = 0
            new_cols.append(name)
        df = df.copy()
        df.columns = new_cols
        return df

    def _read_file(self, uploaded_file, sheet=None) -> pd.DataFrame:
        name = uploaded_file.name.lower()

        if name.endswith(".csv"):
            last_err = None
            for enc in ("utf-8-sig", "latin-1"):
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding=enc)
                    if df.shape[1] == 1:  # maybe ; or tab separated
                        uploaded_file.seek(0)
                        alt = pd.read_csv(uploaded_file, encoding=enc, sep=None, engine="python")
                        if alt.shape[1] > 1:
                            df = alt
                    return df
                except UnicodeDecodeError as e:
                    last_err = e
            raise last_err
        if name.endswith((".xlsx", ".xls")):
            uploaded_file.seek(0)
            return pd.read_excel(uploaded_file, sheet_name=sheet or 0)
        if name.endswith(".json"):
            uploaded_file.seek(0)
            try:
                return pd.read_json(uploaded_file)
            except ValueError:
                uploaded_file.seek(0)
                return pd.read_json(uploaded_file, lines=True)
        if name.endswith(".parquet"):
            uploaded_file.seek(0)
            return pd.read_parquet(uploaded_file)
        raise ValueError(f"Unsupported file type: {uploaded_file.name}")

    # ------------------------------------------------------------------ UI
    def render_upload_ui(self):
        st.header("📂 Upload Dataset")

        uploaded_file = st.file_uploader(
            "Upload Dataset File",
            type=["csv", "xlsx", "xls", "json", "parquet"],
            help="Supports CSV, Excel, JSON, and Parquet files.",
        )

        if uploaded_file is not None:
            sheet = None
            if uploaded_file.name.lower().endswith((".xlsx", ".xls")):
                try:
                    uploaded_file.seek(0)
                    sheets = pd.ExcelFile(uploaded_file).sheet_names
                    uploaded_file.seek(0)
                except Exception as e:
                    st.error(f"❌ Could not open Excel file: {e}")
                    return
                if len(sheets) > 1:
                    sheet = st.selectbox("Select Sheet", sheets, key="excel_sheet")
                else:
                    sheet = sheets[0]

            signature = (uploaded_file.name, uploaded_file.size, sheet)
            if st.session_state.file_sig != signature or st.session_state.df is None:
                try:
                    df = self._read_file(uploaded_file, sheet)
                except Exception as e:
                    st.error(f"❌ Could not read file: {e}")
                    return

                if df is None or df.shape[0] == 0 or df.shape[1] == 0:
                    st.error("❌ The file is empty — no rows or columns were found.")
                    return

                df = self._clean_columns(df)
                st.session_state.df = df.copy()
                st.session_state.original_df = df.copy()
                st.session_state.file_name = uploaded_file.name
                st.session_state.file_sig = signature
                st.session_state.report_items = []
                st.session_state.history = []
                clear_ml_state()
                st.success(f"🎉 '{uploaded_file.name}' loaded — {df.shape[0]:,} rows × {df.shape[1]} columns.")

        if st.session_state.df is not None:
            st.divider()

            top_c1, top_c2, top_c3 = st.columns([2, 1, 1])
            with top_c1:
                st.subheader("📋 Dataset Viewer")
            with top_c2:
                if ui_button("↩️ Undo Last Change", disabled=not st.session_state.history, key="btn_undo"):
                    if st.session_state.history:
                        st.session_state.df = st.session_state.history.pop()
                        flash("↩️ Reverted last change.")
                        st.rerun()
            with top_c3:
                if ui_button("🔄 Reset to Original", disabled=st.session_state.original_df is None,
                             key="btn_reset"):
                    st.session_state.df = st.session_state.original_df.copy()
                    st.session_state.history = []
                    clear_ml_state()
                    flash("🔄 Dataset reset to the original upload.")
                    st.rerun()

            ui_dataframe(st.session_state.df, height=500)

            csv_data = st.session_state.df.to_csv(index=False).encode("utf-8")
            base_name = (st.session_state.file_name or "dataset").rsplit(".", 1)[0]
            ui_download(
                "📥 Download Current Dataset (CSV)",
                data=csv_data,
                file_name=f"processed_{base_name}.csv",
                mime="text/csv",
                key="dl_loader_csv",
            )
