import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from pipeline.theme import (
    AUTO_TEMPLATE, CHART_TEMPLATES, CONTINUOUS_SCALES, QUALITATIVE_PALETTES,
    chart_template, color_swatches_html, get_theme, theme_colors,
)
from pipeline.utils import column_groups, ui_dataframe, ui_download, ui_plot

try:  # needed only for OLS trendlines
    import statsmodels  # noqa: F401
    HAS_STATSMODELS = True
except Exception:  # pragma: no cover
    HAS_STATSMODELS = False

MAX_PLOT_POINTS = 20000
THEME_PALETTE = "🎨 App Theme Colors"
CUSTOM_PALETTE = "✏️ Custom Colors"


def _sample(df: pd.DataFrame, n: int = MAX_PLOT_POINTS) -> pd.DataFrame:
    return df if len(df) <= n else df.sample(n, random_state=42)


def _top_categories(df: pd.DataFrame, col: str, n: int):
    """Keep only rows belonging to the n most frequent categories of `col`."""
    top = df[col].value_counts().head(n).index
    return df[df[col].isin(top)], df[col].nunique() > n


class PowerBIDashboardEngine:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.kd, self.kc, self.kb = {}, {}, {}

    # ------------------------------------------------------------------ colour studio
    def _render_color_studio(self):
        theme = get_theme()
        with st.expander("🎨 Chart Color Studio — choose palette, scale & style", expanded=False):
            c1, c2, c3 = st.columns(3)
            with c1:
                palette_name = st.selectbox(
                    "Category Palette",
                    [THEME_PALETTE] + list(QUALITATIVE_PALETTES.keys()) + [CUSTOM_PALETTE],
                    key="eda_palette",
                    help="Colors used for categories / groups and single-color charts.",
                )
            with c2:
                default_idx = CONTINUOUS_SCALES.index(theme["scale"]) if theme["scale"] in CONTINUOUS_SCALES else 0
                scale = st.selectbox("Continuous Color Scale", CONTINUOUS_SCALES, index=default_idx,
                                     key="eda_cscale", help="Used for heatmaps, correlations and numeric coloring.")
            with c3:
                tmpl_choice = st.selectbox("Chart Style", CHART_TEMPLATES, key="eda_template")
            reverse = st.checkbox("Reverse continuous scale", key="eda_rev")

            if palette_name == THEME_PALETTE:
                seq = theme_colors()
            elif palette_name == CUSTOM_PALETTE:
                defaults = theme_colors()
                pick_cols = st.columns(5)
                seq = []
                for i, pc in enumerate(pick_cols):
                    with pc:
                        seq.append(st.color_picker(f"Color {i + 1}", defaults[i % len(defaults)], key=f"eda_cc{i}"))
            else:
                seq = list(QUALITATIVE_PALETTES[palette_name])

            st.markdown("**Preview**", unsafe_allow_html=False)
            st.markdown(color_swatches_html(seq[:10]), unsafe_allow_html=True)

        template = chart_template() if tmpl_choice == AUTO_TEMPLATE else tmpl_choice
        st.session_state["_transparent_charts"] = tmpl_choice == AUTO_TEMPLATE
        cscale = f"{scale}_r" if reverse else scale

        self.kd = dict(color_discrete_sequence=seq, template=template)
        self.kc = dict(color_continuous_scale=cscale, template=template)
        self.kb = {**self.kd, **self.kc}
        self.seq = seq

    def _plot(self, builder, key: str):
        """Build and draw a figure; show a friendly error instead of crashing the page."""
        try:
            fig = builder()
            ui_plot(fig, key=key)
        except Exception as ex:
            st.error(f"❌ Could not draw this chart: {ex}")

    # ------------------------------------------------------------------ main page
    def render_dashboard(self):
        st.title("📊 Enterprise Exploratory Data Analysis (EDA) Studio")

        if self.df is None or self.df.empty:
            st.error("⚠️ Please Upload Dataset")
            return

        self._render_color_studio()

        full_df = self.df
        filtered_df = full_df
        num_cols, cat_cols, dt_cols = column_groups(full_df)
        all_cols = full_df.columns.tolist()

        # ---------------- slicers
        slicer_c1, slicer_c2 = st.columns(2)
        with slicer_c1:
            if cat_cols:
                filter_cat_col = st.selectbox("Category Filter", ["None"] + cat_cols, key="pbi_cat_slicer")
                if filter_cat_col != "None":
                    try:
                        n_unique = full_df[filter_cat_col].nunique(dropna=True)
                        if n_unique > 200:
                            st.warning(f"`{filter_cat_col}` has {n_unique} categories — too many for a filter.")
                        else:
                            options = sorted(full_df[filter_cat_col].dropna().unique().tolist(), key=str)
                            selected = st.multiselect(f"Filter by {filter_cat_col}", options, default=options,
                                                      key=f"pbi_cat_vals_{filter_cat_col}")
                            if not selected:
                                st.info("No values selected — filter ignored.")
                            elif len(selected) != len(options):
                                filtered_df = filtered_df[filtered_df[filter_cat_col].isin(selected)]
                    except TypeError:
                        st.info("This column holds unhashable values and cannot be used as a filter.")

        with slicer_c2:
            if num_cols:
                filter_num_col = st.selectbox("Numeric Range Filter", ["None"] + num_cols, key="pbi_num_slicer")
                if filter_num_col != "None":
                    s = full_df[filter_num_col].dropna().astype("float64")
                    s = s[np.isfinite(s)]
                    if s.empty:
                        st.info("This column has no numeric values to filter.")
                    elif s.min() == s.max():
                        st.info(f"`{filter_num_col}` is constant ({s.min():g}) — no range to filter.")
                    else:
                        lo, hi = float(s.min()), float(s.max())
                        sel = st.slider(f"Range for {filter_num_col}", lo, hi, (lo, hi),
                                        key=f"pbi_range_{filter_num_col}")
                        if sel != (lo, hi):
                            filtered_df = filtered_df[filtered_df[filter_num_col].between(sel[0], sel[1])]

        st.divider()

        if filtered_df.empty:
            st.warning("⚠️ The current filters leave no rows. Widen the filters to see charts.")
            return

        # ---------------- KPI cards
        kpi = st.columns(4)
        cards = [
            ("Active Rows", f"{len(filtered_df):,}"),
            ("Total Columns", f"{len(all_cols)}"),
            ("Numeric Features", f"{len(num_cols)}"),
            ("Categorical Features", f"{len(cat_cols)}"),
        ]
        for col, (title, value) in zip(kpi, cards):
            with col:
                st.markdown(
                    f'<div class="pbi-card"><div class="pbi-card-title">{title}</div>'
                    f'<div class="pbi-card-value">{value}</div></div>',
                    unsafe_allow_html=True,
                )

        st.divider()

        tab_uni, tab_bi, tab_multi, tab_outlier, tab_missing, tab_custom = st.tabs([
            "🟢 Univariate EDA", "🟡 Bivariate EDA", "🔵 Multivariate EDA",
            "📦 Outliers & Skewness", "🕳️ Missing Value Matrix", "🎨 All Custom Studio",
        ])

        with tab_uni:
            self._tab_univariate(filtered_df, num_cols, cat_cols)
        with tab_bi:
            self._tab_bivariate(filtered_df, num_cols, cat_cols)
        with tab_multi:
            self._tab_multivariate(filtered_df, num_cols, cat_cols, all_cols)
        with tab_outlier:
            self._tab_outliers(filtered_df, num_cols)
        with tab_missing:
            self._tab_missing(filtered_df)
        with tab_custom:
            self._tab_custom(filtered_df, all_cols, num_cols)

        st.divider()
        st.markdown("### 📥 Export Processed Dataset")
        ui_download(
            label="📄 Download Active Dataset (CSV)",
            data=filtered_df.to_csv(index=False).encode("utf-8"),
            file_name="autodatalab_eda_export.csv",
            mime="text/csv",
            key="dl_eda_csv",
        )

    # ------------------------------------------------------------------ tabs
    def _tab_univariate(self, df, num_cols, cat_cols):
        st.subheader("🟢 Single Variable Distribution & Frequency Analysis")
        feature_type = st.radio("Select Feature Type", ["Numerical", "Categorical"], horizontal=True, key="uni_type")

        if feature_type == "Numerical":
            if not num_cols:
                st.info("No numerical features in the dataset.")
                return
            sel = st.selectbox("Select Numerical Feature", num_cols, key="uni_num_sel")
            data = df[sel].astype("float64").replace([np.inf, -np.inf], np.nan)
            if data.dropna().empty:
                st.info("This feature has no numeric values in the current view.")
                return
            c1, c2 = st.columns(2)
            with c1:
                bins = st.slider("Histogram Bins", 10, 100, 30, key="uni_bins")
                self._plot(lambda: px.histogram(df, x=sel, nbins=bins, marginal="rug",
                                                title=f"Histogram & Rug Plot for {sel}", **self.kd), "uni_hist")
            with c2:
                pts = "all" if len(df) <= 5000 else "outliers"
                self._plot(lambda: px.box(df, y=sel, points=pts, title=f"Box Plot for {sel}", **self.kd), "uni_box")
            m = st.columns(4)
            m[0].metric("Mean", f"{data.mean():.2f}")
            m[1].metric("Median", f"{data.median():.2f}")
            m[2].metric("Std Dev", f"{data.std():.2f}")
            m[3].metric("Skewness", f"{data.skew():.2f}")
        else:
            if not cat_cols:
                st.info("No categorical features in the dataset.")
                return
            sel = st.selectbox("Select Categorical Feature", cat_cols, key="uni_cat_sel")
            top_n = st.slider("Top N Categories", 5, 30, 10, key="uni_topn")
            try:
                counts = df[sel].astype(str).where(df[sel].notna()).value_counts().head(top_n).reset_index()
            except Exception as ex:
                st.error(f"❌ Cannot count this column: {ex}")
                return
            counts.columns = [sel, "Count"]
            c1, c2 = st.columns(2)
            with c1:
                self._plot(lambda: px.bar(counts, x=sel, y="Count", color="Count",
                                          title=f"Top {top_n} Bar Chart for {sel}", **self.kc), "uni_bar")
            with c2:
                self._plot(lambda: px.pie(counts, names=sel, values="Count", hole=0.4,
                                          title=f"Distribution Pie Chart for {sel}", **self.kd), "uni_pie")

    def _tab_bivariate(self, df, num_cols, cat_cols):
        st.subheader("🟡 Two-Variable Relationship Analysis")
        mode = st.selectbox("Select Relationship Combination",
                            ["Numerical vs Numerical", "Categorical vs Numerical", "Categorical vs Categorical"],
                            key="bi_mode_sel")

        if mode == "Numerical vs Numerical":
            if len(num_cols) < 2:
                st.info("Need at least 2 numerical features for a scatter plot.")
                return
            c1, c2, c3 = st.columns(3)
            with c1:
                x = st.selectbox("X-Axis (Numeric)", num_cols, index=0, key="bi_num_x")
            with c2:
                y = st.selectbox("Y-Axis (Numeric)", num_cols, index=min(1, len(num_cols) - 1), key="bi_num_y")
            with c3:
                trend = st.checkbox("Add Trendline (OLS)", value=HAS_STATSMODELS, key="bi_ols",
                                    disabled=not HAS_STATSMODELS,
                                    help=None if HAS_STATSMODELS else "Install statsmodels to enable trendlines")
            plot_df = _sample(df)
            if len(plot_df) < len(df):
                st.caption(f"Showing a random sample of {len(plot_df):,} points for speed.")
            self._plot(lambda: px.scatter(plot_df, x=x, y=y, trendline="ols" if (trend and x != y) else None,
                                          title=f"Scatter Plot: {x} vs {y}", **self.kb), "bi_scatter")

        elif mode == "Categorical vs Numerical":
            if not (cat_cols and num_cols):
                st.info("Need both categorical and numerical features.")
                return
            c1, c2, c3 = st.columns(3)
            with c1:
                x_cat = st.selectbox("Category (X-Axis)", cat_cols, key="bi_cat_x")
            with c2:
                y_num = st.selectbox("Numeric Value (Y-Axis)", num_cols, key="bi_num_y2")
            with c3:
                chart_type = st.radio("Visual Type", ["Box Plot", "Violin Plot", "Bar (Mean Aggregation)"],
                                      key="bi_cat_num_vtype")
            sub, truncated = _top_categories(df, x_cat, 20)
            if truncated:
                st.caption("Showing the 20 most frequent categories.")
            sub = _sample(sub)

            def build():
                if chart_type == "Box Plot":
                    return px.box(sub, x=x_cat, y=y_num, color=x_cat, title=f"Boxplot of {y_num} by {x_cat}", **self.kd)
                if chart_type == "Violin Plot":
                    return px.violin(sub, x=x_cat, y=y_num, color=x_cat, box=True,
                                     title=f"Violin Plot of {y_num} by {x_cat}", **self.kd)
                agg = sub.groupby(x_cat, observed=True)[y_num].mean().reset_index()
                return px.bar(agg, x=x_cat, y=y_num, color=y_num, title=f"Mean {y_num} by {x_cat}", **self.kb)
            self._plot(build, "bi_catnum")

        else:
            if len(cat_cols) < 2:
                st.info("Need at least 2 categorical columns.")
                return
            c1, c2 = st.columns(2)
            with c1:
                cat1 = st.selectbox("Primary Category", cat_cols, index=0, key="bi_cat1")
            with c2:
                cat2 = st.selectbox("Grouped Category", cat_cols, index=min(1, len(cat_cols) - 1), key="bi_cat2")
            if cat1 == cat2:
                st.info("Pick two different columns.")
                return
            sub, t1 = _top_categories(df, cat1, 15)
            sub, t2 = _top_categories(sub, cat2, 15)
            if t1 or t2:
                st.caption("Showing the 15 most frequent categories of each column.")
            self._plot(lambda: px.histogram(sub, x=cat1, color=cat2, barmode="group",
                                            title=f"Grouped Bar Chart: {cat1} by {cat2}", **self.kd), "bi_catcat")
            st.markdown("##### Crosstab / Contingency Table")
            ui_dataframe(pd.crosstab(sub[cat1], sub[cat2]))

    def _tab_multivariate(self, df, num_cols, cat_cols, all_cols):
        st.subheader("🔵 Complex Multi-Feature Relationships")
        m1, m2, m3 = st.tabs(["🔥 Correlation Matrix", "🌌 3D Scatter Plot", "🕸️ Pairwise Matrix"])

        with m1:
            if len(num_cols) >= 2:
                method = st.selectbox("Correlation Method", ["pearson", "spearman", "kendall"], key="corr_method")
                data = df[num_cols].astype("float64").replace([np.inf, -np.inf], np.nan)
                if method == "kendall" and len(data) > 5000:
                    data = data.sample(5000, random_state=42)
                    st.caption("Kendall correlation computed on a 5,000-row sample for speed.")
                corr = data.corr(method=method).round(2)
                self._plot(lambda: px.imshow(corr, text_auto=len(num_cols) <= 15, zmin=-1, zmax=1, aspect="auto",
                                             title=f"{method.capitalize()} Correlation Matrix", **self.kc), "multi_corr")
            else:
                st.info("Need at least 2 numerical columns for a correlation matrix.")

        with m2:
            if len(num_cols) >= 3:
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    x3 = st.selectbox("3D X-Axis", num_cols, index=0, key="3d_x")
                with c2:
                    y3 = st.selectbox("3D Y-Axis", num_cols, index=1, key="3d_y")
                with c3:
                    z3 = st.selectbox("3D Z-Axis", num_cols, index=2, key="3d_z")
                with c4:
                    color3 = st.selectbox("Color By", [None] + all_cols, key="3d_color")
                plot_df = _sample(df, 10000)
                self._plot(lambda: px.scatter_3d(plot_df, x=x3, y=y3, z=z3, color=color3,
                                                 title="3D Multivariate Scatter", **self.kb), "multi_3d")
            else:
                st.info("Need at least 3 numerical features for a 3D scatter plot.")

        with m3:
            if len(num_cols) >= 2:
                chosen = st.multiselect("Select Features for Pair Plot (max 6)", num_cols,
                                        default=num_cols[:min(4, len(num_cols))], key="pair_cols", max_selections=6)
                hue = st.selectbox("Hue (Color By Category)", [None] + cat_cols, key="pair_hue")
                if len(chosen) >= 2:
                    plot_df = _sample(df, 3000)
                    if hue is not None:
                        plot_df, _ = _top_categories(plot_df, hue, 12)
                    self._plot(lambda: px.scatter_matrix(plot_df, dimensions=chosen, color=hue,
                                                         title="Scatter Matrix (Pair Plot)", **self.kb), "multi_pair")
                else:
                    st.info("Select at least 2 features.")
            else:
                st.info("Need at least 2 columns for a scatter matrix.")

    def _tab_outliers(self, df, num_cols):
        st.subheader("📦 Statistical Anomalies & Distribution Shapes")
        if not num_cols:
            st.info("No numeric columns in the dataset.")
            return
        col = st.selectbox("Select Target Numerical Feature", num_cols, key="outlier_col_target")
        data = df[col].astype("float64").replace([np.inf, -np.inf], np.nan).dropna()
        if data.empty:
            st.info("This feature has no numeric values in the current view.")
            return
        q1, q3 = data.quantile(0.25), data.quantile(0.75)
        iqr = q3 - q1
        lower_b, upper_b = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = data[(data < lower_b) | (data > upper_b)]

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### 📏 Statistical Metrics")
            st.write(f"**Skewness Score:** `{data.skew():.3f}`")
            st.write(f"**Kurtosis Score:** `{data.kurt():.3f}`")
            st.write(f"**Lower Bound (IQR 1.5):** `{lower_b:.2f}`")
            st.write(f"**Upper Bound (IQR 1.5):** `{upper_b:.2f}`")
            st.write(f"**Total Outliers Count:** `{len(outliers)}` (`{len(outliers) / len(data) * 100:.2f}%`)")
        with c2:
            pts = "all" if len(df) <= 5000 else "outliers"
            self._plot(lambda: px.box(df, y=col, points=pts, title=f"Outliers Distribution - {col}", **self.kd),
                       "out_box")

    def _tab_missing(self, df):
        st.subheader("🕳️ Missing Value Matrix")
        if df.isnull().sum().sum() == 0:
            st.success("🎉 No missing values in the current view!")
            return
        view = df
        if len(view) > 1500:
            idx = np.linspace(0, len(view) - 1, 1500).astype(int)
            view = view.iloc[idx]
            st.caption("Matrix downsampled to 1,500 evenly spaced rows for speed.")
        mask = view.isnull().astype(int)
        scale = ["#e2e8f0", self.seq[0]] if self.seq else ["#e2e8f0", "#ef4444"]

        def matrix():
            fig = px.imshow(mask.T, color_continuous_scale=scale, aspect="auto",
                            title="Missing Value Matrix (colored = missing)",
                            labels=dict(x="Row Position", y="Column", color="Missing"),
                            template=self.kd["template"])
            fig.update_layout(coloraxis_showscale=False)
            return fig
        self._plot(matrix, "miss_matrix")

        by_col = df.isnull().sum().sort_values(ascending=False)
        by_col = by_col[by_col > 0]
        self._plot(lambda: px.bar(x=by_col.values, y=by_col.index, orientation="h",
                                  title="Missing Values by Column", labels={"x": "Missing Count", "y": "Column"},
                                  **self.kd), "miss_bar")

    def _tab_custom(self, df, all_cols, num_cols):
        st.subheader("🎨 Custom Dynamic Plotting Engine")
        agg_funcs = {"None (raw values)": None, "Mean": "mean", "Sum": "sum", "Median": "median",
                     "Min": "min", "Max": "max", "Count": "count"}

        def canvas(panel_id: str):
            with st.expander(f"⚙️ Custom Chart Setup ({panel_id.upper()})", expanded=True):
                v_type = st.selectbox(
                    "Chart Type",
                    ["Bar Chart", "Line Chart", "Scatter Plot", "Pie Chart", "Donut Chart", "Box Plot", "Histogram",
                     "Violin Plot", "Density Heatmap"], key=f"{panel_id}_vtype")
                cx, cy = st.columns(2)
                with cx:
                    x_axis = st.selectbox("X-Axis Feature", all_cols, key=f"{panel_id}_xaxis")
                with cy:
                    y_axis = st.selectbox("Y-Axis Feature (Optional)", [None] + num_cols, key=f"{panel_id}_yaxis")
                c_grp, c_agg = st.columns(2)
                with c_grp:
                    group = st.selectbox("Color / Grouping Feature", [None] + all_cols, key=f"{panel_id}_grp")
                with c_agg:
                    agg_name = st.selectbox("Aggregation (Bar / Line)", list(agg_funcs.keys()), key=f"{panel_id}_agg")
                c_title, c_col = st.columns([3, 1])
                with c_title:
                    title = st.text_input("Chart Title (optional)", key=f"{panel_id}_title")
                with c_col:
                    use_custom = st.checkbox("Single color", key=f"{panel_id}_usecol")
                    custom_color = st.color_picker("Pick", self.seq[0] if self.seq else "#38BDF8",
                                                   key=f"{panel_id}_color", disabled=not use_custom,
                                                   label_visibility="collapsed")

            kd = dict(self.kd)
            kb = dict(self.kb)
            if use_custom and group is None:
                kd["color_discrete_sequence"] = [custom_color]
                kb["color_discrete_sequence"] = [custom_color]
            ttl = title or None

            def build():
                data = df
                agg = agg_funcs[agg_name]
                if v_type in ("Bar Chart", "Line Chart"):
                    if y_axis is None and v_type == "Line Chart":
                        raise ValueError("Line charts need a Y-axis feature.")
                    if y_axis is None:
                        return px.histogram(_sample(data), x=x_axis, color=group, title=ttl, **kd)
                    if agg is not None:
                        keys = [x_axis] + ([group] if group and group != x_axis else [])
                        g = data.groupby(keys, observed=True, dropna=True)[y_axis].agg(agg).reset_index()
                        g = g.sort_values(x_axis) if v_type == "Line Chart" else g.sort_values(y_axis, ascending=False).head(50)
                        data = g
                    else:
                        data = _sample(data)
                    if v_type == "Bar Chart":
                        return px.bar(data, x=x_axis, y=y_axis, color=group, title=ttl, **kb)
                    return px.line(data.sort_values(x_axis), x=x_axis, y=y_axis, color=group, title=ttl, **kd)
                if v_type == "Scatter Plot":
                    if y_axis is None:
                        raise ValueError("Scatter plots need a Y-axis feature.")
                    return px.scatter(_sample(data), x=x_axis, y=y_axis, color=group, title=ttl, **kb)
                if v_type in ("Pie Chart", "Donut Chart"):
                    if y_axis is None:
                        pie_df = data[x_axis].astype(str).value_counts().head(15).reset_index()
                        pie_df.columns = [x_axis, "Count"]
                        vals = "Count"
                    else:
                        pie_df = data.groupby(x_axis, observed=True)[y_axis].sum().sort_values(ascending=False).head(15).reset_index()
                        vals = y_axis
                    return px.pie(pie_df, names=x_axis, values=vals, hole=0.45 if v_type == "Donut Chart" else 0,
                                  title=ttl, **kd)
                if v_type == "Box Plot":
                    if y_axis is None:
                        raise ValueError("Box plots need a Y-axis feature.")
                    return px.box(_sample(data), x=x_axis, y=y_axis, color=group, title=ttl, **kd)
                if v_type == "Histogram":
                    return px.histogram(_sample(data), x=x_axis, color=group, marginal="box", title=ttl, **kd)
                if v_type == "Violin Plot":
                    if y_axis is None:
                        raise ValueError("Violin plots need a Y-axis feature.")
                    return px.violin(_sample(data), x=x_axis, y=y_axis, color=group, box=True, title=ttl, **kd)
                if y_axis is None:
                    raise ValueError("Density heatmaps need a Y-axis feature.")
                return px.density_heatmap(_sample(data), x=x_axis, y=y_axis, title=ttl, **self.kc)

            fig_key = f"{panel_id}_fig"
            self._plot(build, fig_key)

        layout = st.radio("Layout Options", ["Single Full Canvas", "2 Columns Dual View"], horizontal=True,
                          key="custom_layout")
        if layout == "Single Full Canvas":
            canvas("custom_canvas_1")
        else:
            left, right = st.columns(2)
            with left:
                canvas("canvas_left")
            with right:
                canvas("canvas_right")
