import math
import warnings

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import (
    LabelEncoder, MinMaxScaler, PolynomialFeatures, RobustScaler, StandardScaler,
)

from pipeline.utils import column_groups, flash, push_history, ui_button

MAX_ONEHOT_NEW_COLUMNS = 500
MAX_POLY_FEATURES = 300


class FeatureEngineeringEngine:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def render_interface(self):
        st.header("⚙️ Feature Engineering & Preprocessing Engine")

        if self.df.empty:
            st.warning("⚠️ The dataset is empty.")
            return

        _labels = ["🔢 Feature Scaling", "🏷️ Categorical Encoding", "📐 Polynomial Features",
            "📅 Date Features", "📦 Binning", "🧮 Custom Formula"]
        _choice = st.radio("Section", _labels, horizontal=True, key="fe_section",
                           label_visibility="collapsed")
        _section = _labels.index(_choice)

        num_cols, cat_cols, _ = column_groups(self.df)

        # TAB 1: FEATURE SCALING
        if _section == 0:
            st.subheader("Scale Numerical Features")
            if num_cols:
                selected = st.multiselect("Select Columns to Scale", num_cols, key="fe_scale_cols")
                scaler_type = st.selectbox("Select Scaler Algorithm",
                                           ["StandardScaler", "MinMaxScaler", "RobustScaler"], key="fe_scaler_type")
                if ui_button("Apply Scaling Pipeline", key="btn_scale"):
                    if not selected:
                        st.warning("⚠️ Please select at least one column.")
                    else:
                        try:
                            scaler = {"StandardScaler": StandardScaler, "MinMaxScaler": MinMaxScaler,
                                      "RobustScaler": RobustScaler}[scaler_type]()
                            data = self.df[selected].astype("float64")
                            if np.isinf(data.to_numpy()).any():
                                raise ValueError("selected columns contain infinite values")
                            scaled = scaler.fit_transform(data)  # NaNs are preserved
                            push_history(self.df)
                            for i, col in enumerate(selected):
                                self.df[col] = scaled[:, i]
                            st.session_state.df = self.df
                            flash(f"✅ Scaled {len(selected)} feature(s) using {scaler_type}!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Scaling failed: {e}")
            else:
                st.info("No numerical columns available for scaling.")

        # TAB 2: CATEGORICAL ENCODING
        if _section == 1:
            st.subheader("Encode Categorical Features")
            if cat_cols:
                encode_cols = st.multiselect("Select Categorical Column(s)", cat_cols, key="fe_enc_cols",
                                             default=cat_cols[:1])
                enc_method = st.radio("Encoding Type", ["Label Encoding", "One-Hot Encoding"], key="fe_enc_method",
                                      horizontal=True)
                new_cols_estimate = 0
                for c in encode_cols:
                    try:
                        n_unique = int(self.df[c].nunique(dropna=True))
                    except TypeError:
                        n_unique = int(self.df[c].astype(str).nunique())
                    new_cols_estimate += max(0, n_unique - 1)
                    if enc_method == "One-Hot Encoding" and n_unique > 50:
                        st.warning(f"⚠️ `{c}` has {n_unique} unique values — one-hot will add many columns.")
                if ui_button("Apply Encoding", key="btn_enc"):
                    if not encode_cols:
                        st.warning("⚠️ Select at least one column.")
                    elif enc_method == "One-Hot Encoding" and new_cols_estimate > MAX_ONEHOT_NEW_COLUMNS:
                        st.error(f"❌ This would create about {new_cols_estimate} new columns "
                                 f"(limit {MAX_ONEHOT_NEW_COLUMNS}). Pick fewer / lower-cardinality columns.")
                    else:
                        try:
                            new_df = self.df.copy()
                            if enc_method == "Label Encoding":
                                for c in encode_cols:
                                    s = new_df[c]
                                    mask = s.notna()
                                    le = LabelEncoder()
                                    codes = pd.Series(pd.NA, index=new_df.index, dtype="Int64")
                                    codes[mask] = le.fit_transform(s[mask].astype(str))
                                    new_df[c] = codes
                            else:
                                new_df = pd.get_dummies(new_df, columns=encode_cols, drop_first=True, dtype=int)
                            push_history(self.df)
                            self.df = new_df
                            st.session_state.df = self.df
                            flash(f"✅ Applied {enc_method} on {len(encode_cols)} column(s)!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Encoding failed: {e}")
            else:
                st.info("No categorical columns remaining for encoding.")

        # TAB 3: POLYNOMIAL FEATURES
        if _section == 2:
            st.subheader("Generate Interaction & Polynomial Features")
            if len(num_cols) >= 2:
                poly_cols = st.multiselect("Select Base Columns (at least 2)", num_cols, key="fe_poly_cols")
                degree = st.slider("Polynomial Degree", 2, 4, 2, key="fe_poly_deg")
                if len(poly_cols) >= 2:
                    n_out = math.comb(len(poly_cols) + degree, degree) - 1
                    st.caption(f"This will create {n_out} features from {len(poly_cols)} columns.")
                if ui_button("Generate Polynomial Features", key="btn_poly"):
                    if len(poly_cols) < 2:
                        st.warning("⚠️ Select at least 2 columns.")
                    elif math.comb(len(poly_cols) + degree, degree) - 1 > MAX_POLY_FEATURES:
                        st.error(f"❌ Too many output features (limit {MAX_POLY_FEATURES}). "
                                 "Choose fewer columns or a lower degree.")
                    elif self.df[poly_cols].isna().any().any():
                        st.error("❌ Selected columns contain missing values. Clean them first (Data Cleaner tab).")
                    else:
                        try:
                            poly = PolynomialFeatures(degree=degree, include_bias=False)
                            arr = poly.fit_transform(self.df[poly_cols].astype("float64"))
                            names = [n.replace(" ", "*") for n in poly.get_feature_names_out(poly_cols)]
                            poly_df = pd.DataFrame(arr, columns=names, index=self.df.index)
                            new_df = pd.concat([self.df.drop(columns=poly_cols), poly_df], axis=1)
                            if new_df.columns.duplicated().any():
                                raise ValueError("generated column names clash with existing columns")
                            push_history(self.df)
                            self.df = new_df
                            st.session_state.df = self.df
                            flash(f"✅ Created {arr.shape[1]} polynomial features!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Polynomial feature generation failed: {e}")
            else:
                st.info("Need at least 2 numerical columns for polynomial features.")

        # TAB 4: DATE FEATURES
        if _section == 3:
            st.subheader("Extract Date/Time Features")
            date_col = st.selectbox("Select Column to Parse as Date", self.df.columns.tolist(), key="fe_date_col")
            if date_col in num_cols:
                st.caption("⚠️ This column is numeric — numbers will be read as timestamps (nanoseconds).")
            parts = st.multiselect(
                "Parts to Extract",
                ["year", "month", "day", "dayofweek", "is_weekend", "hour", "quarter"],
                default=["year", "month", "dayofweek"], key="fe_date_parts",
            )
            if ui_button("Extract Date Features", key="btn_date"):
                if not parts:
                    st.warning("⚠️ Select at least one part to extract.")
                else:
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            parsed = pd.to_datetime(self.df[date_col], errors="coerce")
                        if parsed.isna().all():
                            st.error(f"❌ Column `{date_col}` could not be parsed as dates.")
                        else:
                            new_df = self.df.copy()
                            ok = parsed.notna()
                            extractors = {
                                "year": lambda p: p.dt.year,
                                "month": lambda p: p.dt.month,
                                "day": lambda p: p.dt.day,
                                "dayofweek": lambda p: p.dt.dayofweek,
                                "is_weekend": lambda p: (p.dt.dayofweek >= 5).astype("float64").where(ok),
                                "hour": lambda p: p.dt.hour,
                                "quarter": lambda p: p.dt.quarter,
                            }
                            for part in parts:
                                new_df[f"{date_col}_{part}"] = extractors[part](parsed).astype("Int64")
                            push_history(self.df)
                            self.df = new_df
                            st.session_state.df = self.df
                            n_bad = int((~ok).sum()) - int(self.df[date_col].isna().sum())
                            msg = f"✅ Extracted {len(parts)} date feature(s) from `{date_col}`!"
                            if n_bad > 0:
                                msg += f" ⚠️ {n_bad} value(s) were not valid dates."
                            flash(msg)
                            st.rerun()
                    except Exception as e:
                        st.error(f"❌ Failed to extract date features: {e}")

        # TAB 5: BINNING
        if _section == 4:
            st.subheader("Bin / Discretize a Numerical Column")
            if num_cols:
                bin_col = st.selectbox("Select Column to Bin", num_cols, key="fe_bin_col")
                n_bins = st.slider("Number of Bins", 2, 20, 5, key="fe_bin_n")
                bin_strategy = st.radio("Binning Strategy", ["Equal Width", "Equal Frequency (Quantile)"],
                                        key="fe_bin_strat", horizontal=True)
                new_col_name = st.text_input("New Column Name", value=f"{bin_col}_binned",
                                             key=f"fe_bin_name_{bin_col}").strip()
                if ui_button("Apply Binning", key="btn_bin"):
                    if not new_col_name:
                        st.warning("⚠️ Enter a name for the new column.")
                    elif self.df[bin_col].nunique(dropna=True) < 2:
                        st.error("❌ This column has fewer than 2 distinct values — nothing to bin.")
                    else:
                        try:
                            s = self.df[bin_col].astype("float64")
                            if bin_strategy == "Equal Width":
                                b = pd.cut(s, bins=n_bins)
                            else:
                                b = pd.qcut(s, q=n_bins, duplicates="drop")
                            push_history(self.df)
                            self.df[new_col_name] = b.astype(str).where(b.notna())
                            st.session_state.df = self.df
                            flash(f"✅ Created `{new_col_name}` from `{bin_col}` using {bin_strategy} binning!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Binning failed: {e}")
            else:
                st.info("No numerical columns available for binning.")

        # TAB 6: CUSTOM FORMULA
        if _section == 5:
            st.subheader("Create a Custom Feature")
            st.caption("Reference columns by name, e.g. `price * quantity` or `income / (age + 1)`. "
                       "Wrap names containing spaces in backticks: `` `unit price` * qty ``.")
            st.write("Available numeric columns:", ", ".join(f"`{c}`" for c in num_cols) or "None")

            new_feature_name = st.text_input("New Feature Name", value="new_feature", key="fe_formula_name").strip()
            formula = st.text_input("Formula", placeholder="e.g. col_a * col_b - col_c", key="fe_formula_expr")

            if ui_button("Create Feature", key="btn_formula"):
                if not formula.strip():
                    st.warning("⚠️ Please enter a formula.")
                elif not new_feature_name:
                    st.warning("⚠️ Please enter a feature name.")
                elif "@" in formula or "__" in formula:
                    st.error("❌ '@' and '__' are not allowed in formulas.")
                else:
                    try:
                        result = self.df.eval(formula.strip())
                        if isinstance(result, pd.DataFrame) or not isinstance(
                                result, (pd.Series, np.ndarray, int, float, bool, np.number, np.bool_)):
                            raise ValueError("formula must be a single expression (no assignments)")
                        if isinstance(result, np.ndarray) and result.shape[0] != len(self.df):
                            raise ValueError("formula result has the wrong length")
                        new_df = self.df.copy()
                        new_df[new_feature_name] = result
                        n_inf = 0
                        if pd.api.types.is_numeric_dtype(new_df[new_feature_name]):
                            col = new_df[new_feature_name].astype("float64")
                            n_inf = int(np.isinf(col).sum())
                            if n_inf:
                                new_df[new_feature_name] = col.replace([np.inf, -np.inf], np.nan)
                        push_history(self.df)
                        self.df = new_df
                        st.session_state.df = self.df
                        msg = f"✅ Created new feature `{new_feature_name}`!"
                        if n_inf:
                            msg += f" ⚠️ {n_inf} infinite value(s) (e.g. divide by zero) were set to missing."
                        flash(msg)
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Formula error: {e}")
