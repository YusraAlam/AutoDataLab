import streamlit as st
import pandas as pd
import plotly.express as px

from pipeline.theme import chart_template, get_theme
from pipeline.utils import column_groups, safe_duplicate_count, ui_dataframe, ui_plot


class DataExplorerEngine:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def render_explorer_interface(self):
        st.subheader("🔍 Interactive Data Explorer")
        st.markdown("---")

        n_rows = len(self.df)
        if n_rows == 0 or self.df.shape[1] == 0:
            st.warning("⚠️ The dataset is empty.")
            return

        template = chart_template()
        scale = get_theme()["scale"]
        num_cols, cat_cols, _ = column_groups(self.df)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Rows", f"{n_rows:,}")
        col2.metric("Total Columns", self.df.shape[1])
        col3.metric("Duplicate Rows", f"{safe_duplicate_count(self.df):,}")
        col4.metric("Total Missing Values", f"{int(self.df.isna().sum().sum()):,}")

        st.markdown("---")

        st.subheader("📋 Dataset Preview")
        if n_rows <= 5:
            rows_to_show = n_rows
        else:
            rows_to_show = st.slider("How many rows do you want to see", min_value=5, max_value=n_rows,
                                     value=min(10, n_rows), key="exp_rows")
        ui_dataframe(self.df.head(rows_to_show))

        st.subheader("📑 Column Summary & Data Types")
        try:
            nunique = self.df.nunique()
        except TypeError:  # unhashable cells
            nunique = self.df.astype(str).nunique()
        col_summary = pd.DataFrame({
            "Data Type": self.df.dtypes.astype(str),
            "Non-Null Count": self.df.notnull().sum(),
            "Missing Values": self.df.isnull().sum(),
            "Missing Ratio (%)": ((self.df.isnull().sum() / n_rows) * 100).round(2),
            "Unique Values": nunique,
        })
        ui_dataframe(
            col_summary,
            column_config={
                "Missing Ratio (%)": st.column_config.ProgressColumn(
                    "Missing Ratio (%)", format="%.2f%%", min_value=0, max_value=100
                )
            },
        )

        st.subheader("📊 Statistical Summary")
        tab1, tab2, tab3, tab4 = st.tabs([
            "🔢 Numerical Columns", "🔤 Categorical Columns",
            "🔥 Correlation Heatmap", "📈 Column Distribution",
        ])

        with tab1:
            if num_cols:
                stats_df = self.df[num_cols].astype("float64").describe().T
                ui_dataframe(stats_df.round(3))
            else:
                st.info("Dataset has no numerical columns.")

        with tab2:
            if cat_cols:
                try:
                    cat_stats_df = self.df[cat_cols].describe().T
                except Exception:
                    cat_stats_df = self.df[cat_cols].astype(str).describe().T
                ui_dataframe(cat_stats_df)
            else:
                st.info("Dataset has no categorical columns.")

        with tab3:
            if len(num_cols) >= 2:
                method = st.selectbox("Correlation Method", ["pearson", "spearman", "kendall"], key="exp_corr_method")
                data = self.df[num_cols].astype("float64")
                if method == "kendall" and len(data) > 5000:
                    data = data.sample(5000, random_state=42)
                    st.caption("Kendall correlation computed on a 5,000-row sample for speed.")
                corr = data.corr(method=method).round(2)
                fig = px.imshow(corr, text_auto=len(num_cols) <= 15, color_continuous_scale="RdBu_r",
                                zmin=-1, zmax=1, title=f"{method.capitalize()} Correlation Heatmap",
                                aspect="auto", template=template)
                ui_plot(fig, key="exp_corr_fig")
            else:
                st.info("Need at least 2 numerical columns for a correlation heatmap.")

        with tab4:
            selected_col = st.selectbox("Select a column to inspect its distribution",
                                        self.df.columns.tolist(), key="exp_dist_col")
            series = self.df[selected_col]
            try:
                if selected_col in num_cols:
                    if series.dropna().empty:
                        st.info("This column only contains missing values.")
                    else:
                        c1, c2 = st.columns(2)
                        with c1:
                            fig_h = px.histogram(self.df, x=selected_col, marginal="rug",
                                                 title=f"Histogram of {selected_col}", template=template,
                                                 color_discrete_sequence=[get_theme()["accent"]])
                            ui_plot(fig_h, key="exp_hist")
                        with c2:
                            fig_b = px.box(self.df, y=selected_col,
                                           points="all" if n_rows <= 5000 else "outliers",
                                           title=f"Box Plot of {selected_col}", template=template,
                                           color_discrete_sequence=[get_theme()["accent2"]])
                            ui_plot(fig_b, key="exp_box")
                else:
                    counts = series.astype(str).where(series.notna()).value_counts().head(20).reset_index()
                    counts.columns = [selected_col, "Count"]
                    fig = px.bar(counts, x=selected_col, y="Count", color="Count",
                                 color_continuous_scale=scale,
                                 title=f"Top values in {selected_col}", template=template)
                    ui_plot(fig, key="exp_bar")
            except Exception as e:
                st.error(f"❌ Could not draw this chart: {e}")
