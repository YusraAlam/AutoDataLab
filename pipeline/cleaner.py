import math

import numpy as np
import pandas as pd
import streamlit as st

from pipeline.utils import (
    column_groups, drop_last_history, flash, push_history, safe_duplicate_count,
    safe_duplicated, ui_button,
)


class DataCleanerEngine:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    # ------------------------------------------------------------------ core methods
    def handle_missing_values(self, columns, strategy: str, constant_val=None):
        """Apply a missing-value strategy. Returns (df, skipped_messages)."""
        skipped = []

        if strategy == "Drop Rows":
            self.df = self.df.dropna(subset=list(columns))
            return self.df, skipped

        for column in columns:
            s = self.df[column]
            is_numeric = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
            is_int = pd.api.types.is_integer_dtype(s)

            if strategy in ("Mean", "Median"):
                if not is_numeric:
                    skipped.append(f"`{column}` is not numeric — {strategy} skipped")
                    continue

                fill = s.mean() if strategy == "Mean" else s.median()

                if pd.isna(fill):
                    skipped.append(f"`{column}` has no values to compute {strategy}")
                    continue

                if is_int:
                    s = s.astype("float64")

                self.df[column] = s.fillna(fill)

            elif strategy == "Mode":
                mode_val = s.mode(dropna=True)

                if mode_val.empty:
                    skipped.append(f"`{column}` has no mode (all missing)")
                    continue

                self.df[column] = s.fillna(mode_val.iloc[0])

            elif strategy == "Constant":
                cval = constant_val

                if is_numeric:
                    try:
                        num = float(cval)
                    except (TypeError, ValueError):
                        skipped.append(
                            f"`{column}` is numeric but '{cval}' is not a number"
                        )
                        continue

                    if is_int and not num.is_integer():
                        s = s.astype("float64")
                    elif is_int:
                        num = int(num)

                    self.df[column] = s.fillna(num)

                elif pd.api.types.is_datetime64_any_dtype(s):
                    parsed = pd.to_datetime(cval, errors="coerce")

                    if pd.isna(parsed):
                        skipped.append(
                            f"`{column}` is a date column but '{cval}' is not a valid date"
                        )
                        continue

                    self.df[column] = s.fillna(parsed)

                else:
                    if isinstance(s.dtype, pd.CategoricalDtype) and cval not in s.cat.categories:
                        s = s.cat.add_categories([cval])

                    self.df[column] = s.fillna(cval)

            elif strategy == "Forward Fill (ffill)":
                self.df[column] = s.ffill()

            elif strategy == "Backward Fill (bfill)":
                self.df[column] = s.bfill()

        return self.df, skipped

    def remove_duplicates(self) -> int:
        initial_count = len(self.df)

        mask = safe_duplicated(self.df)

        self.df = self.df.loc[~mask].reset_index(drop=True)

        return initial_count - len(self.df)

    @staticmethod
    def _convert_series(s: pd.Series, target_type: str) -> pd.Series:
        if target_type == "int":
            num = pd.to_numeric(s, errors="coerce")

            if np.isinf(num.astype("float64")).any():
                raise ValueError("column contains infinite values")

            if ((num.dropna() % 1) != 0).any():
                raise ValueError(
                    "column contains decimals — convert to float instead (or round first)"
                )

            return num.astype("Int64")

        if target_type == "float":
            return pd.to_numeric(s, errors="coerce").astype("float64")

        if target_type == "string":
            return s.where(s.isna(), s.astype(str))

        if target_type == "datetime":
            return pd.to_datetime(s, errors="coerce")

        if target_type == "category":
            return s.astype("category")

        raise ValueError(f"Unknown target type: {target_type}")

    def convert_datatype(self, column: str, target_type: str):
        """Convert a column. Returns (df, newly_missing_count)."""
        before_na = int(self.df[column].isna().sum())

        new_series = self._convert_series(
            self.df[column],
            target_type
        )

        after_na = int(new_series.isna().sum())

        n_valid_before = len(self.df) - before_na

        if (
            target_type in ("int", "float", "datetime")
            and n_valid_before > 0
            and after_na - before_na >= n_valid_before
        ):
            raise ValueError(
                f"none of the values could be converted to {target_type}"
            )

        self.df[column] = new_series

        return self.df, max(0, after_na - before_na)

    def cap_outliers(self, column: str, method: str = "IQR"):
        """Winsorise a column. Returns (df, capped_count)."""
        s = self.df[column]

        if method == "IQR":
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

        else:
            mean, std = s.mean(), s.std()

            if pd.isna(std) or std == 0:
                return self.df, 0

            lower = mean - 3 * std
            upper = mean + 3 * std

        if pd.isna(lower) or pd.isna(upper):
            return self.df, 0

        if pd.api.types.is_integer_dtype(s):
            lower = math.ceil(lower)
            upper = math.floor(upper)

        capped = int(((s < lower) | (s > upper)).sum())

        self.df[column] = s.clip(
            lower=lower,
            upper=upper
        )

        return self.df, capped

    # ------------------------------------------------------------------ UI
    def render_cleaning_interface(self):
        st.header("Data Cleaning")

        st.caption(
            "Use the Undo button (Data Loader tab) any time to step back one change."
        )

        if self.df.empty:
            st.warning("The dataset is empty.")
            return

        _labels = [
            "Missing Values",
            "Duplicates",
            "Data Types",
            "Outliers"
        ]

        _choice = st.radio(
            "Section",
            _labels,
            horizontal=True,
            key="clean_section",
            label_visibility="collapsed"
        )

        _section = _labels.index(_choice)

        # TAB 1: MISSING VALUES
        if _section == 0:
            st.subheader("Missing Value Treatment")

            missing_cols = self.df.columns[
                self.df.isnull().any()
            ].tolist()

            if not missing_cols:
                st.success("Data has no missing values")

            else:
                cols = st.multiselect(
                    "Select Target Column(s)",
                    missing_cols,
                    key="clean_cols",
                    default=missing_cols[:1]
                )

                total_rows = len(self.df)

                for col in cols:
                    missing_count = int(
                        self.df[col].isnull().sum()
                    )

                    missing_pct = (
                        missing_count / total_rows * 100
                    )

                    msg = (
                        f"`{col}` ({self.df[col].dtype}): "
                        f"**{missing_pct:.2f}%** missing "
                        f"({missing_count}/{total_rows})."
                    )

                    (
                        st.warning
                        if missing_pct > 5.0
                        else st.info
                    )(msg)

                strat = st.selectbox(
                    "Strategy (applied to all selected columns)",
                    [
                        "Mean",
                        "Median",
                        "Mode",
                        "Constant",
                        "Forward Fill (ffill)",
                        "Backward Fill (bfill)",
                        "Drop Rows"
                    ],
                    key="clean_strat",
                )

                c_val = None

                if strat == "Constant":
                    c_val = st.text_input(
                        "Constant Value",
                        placeholder="e.g. 0 or Unknown",
                        key="clean_const"
                    )

                needs_value = (
                    strat == "Constant"
                    and not (c_val or "").strip()
                )

                if needs_value:
                    st.caption(
                        "Enter a constant value to enable the button."
                    )

                if ui_button(
                    "Apply Cleaning Operation",
                    key="btn_missing",
                    disabled=(not cols or needs_value)
                ):
                    try:
                        before_missing = int(
                            self.df[cols].isnull().sum().sum()
                        )

                        before_rows = len(self.df)

                        push_history(self.df)

                        self.df, skipped = self.handle_missing_values(
                            cols,
                            strat,
                            (c_val or "").strip()
                        )

                        after_missing = int(
                            self.df[cols].isnull().sum().sum()
                        )

                        changed = (
                            before_missing != after_missing
                            or len(self.df) != before_rows
                        )

                        if not changed:
                            drop_last_history()

                            warning_msg = "Nothing changed."

                            if skipped:
                                warning_msg += " " + "; ".join(skipped)

                            st.warning(warning_msg)

                        else:
                            st.session_state.df = self.df

                            msg = (
                                f"Applied {strat} to "
                                f"{len(cols)} column(s)."
                            )

                            if len(self.df) != before_rows:
                                msg += (
                                    f" {before_rows - len(self.df)} "
                                    f"row(s) dropped."
                                )

                            if skipped:
                                msg += (
                                    " Skipped: "
                                    + "; ".join(skipped)
                                )

                            flash(msg)

                            st.rerun()

                    except Exception as e:
                        drop_last_history()

                        st.error(
                            f"Missing-value treatment failed: {e}"
                        )

        # TAB 2: DUPLICATES
        if _section == 1:
            st.subheader("Duplicate Records")

            dups = safe_duplicate_count(self.df)

            st.metric(
                "Total Duplicate Rows",
                dups
            )

            if dups > 0:
                if ui_button(
                    "Remove Duplicates Now",
                    key="btn_dups"
                ):
                    push_history(self.df)

                    removed_count = self.remove_duplicates()

                    st.session_state.df = self.df

                    flash(
                        f"{removed_count} duplicate rows removed."
                    )

                    st.rerun()

            else:
                st.info("No duplicates found.")

        # TAB 3: DATA TYPES
        if _section == 2:
            st.subheader("Data Type Conversion")

            col_target = st.selectbox(
                "Select Column to Convert",
                self.df.columns,
                key="dt_col"
            )

            st.write(
                f"Current Type: `{self.df[col_target].dtype}`"
            )

            target_type = st.selectbox(
                "Convert To",
                [
                    "int",
                    "float",
                    "string",
                    "datetime",
                    "category"
                ],
                key="dt_type"
            )

            if ui_button(
                "Convert Data Type",
                key="btn_dt"
            ):
                try:
                    push_history(self.df)

                    self.df, new_na = self.convert_datatype(
                        col_target,
                        target_type
                    )

                    st.session_state.df = self.df

                    msg = (
                        f"Column `{col_target}` converted "
                        f"to `{target_type}`!"
                    )

                    if new_na:
                        msg += (
                            f" {new_na} value(s) could not be "
                            f"converted and became missing."
                        )

                    flash(msg)

                    st.rerun()

                except Exception as e:
                    drop_last_history()

                    self.df = st.session_state.df.copy()

                    st.error(
                        f"Type conversion failed: {e}"
                    )

        # TAB 4: OUTLIERS
        if _section == 3:
            st.subheader("Outlier Capping Engine")

            num_cols, _, _ = column_groups(self.df)

            if num_cols:
                out_col = st.selectbox(
                    "Select Numerical Column",
                    num_cols,
                    key="out_col"
                )

                method = st.radio(
                    "Capping Method",
                    ["IQR", "Z-Score"],
                    key="out_method",
                    horizontal=True
                )

                if ui_button(
                    "Cap Outliers",
                    key="btn_out"
                ):
                    try:
                        push_history(self.df)

                        self.df, capped = self.cap_outliers(
                            out_col,
                            method=method
                        )

                        if capped == 0:
                            drop_last_history()

                            st.info(
                                f"No outliers found in `{out_col}` "
                                f"using {method} bounds."
                            )

                        else:
                            st.session_state.df = self.df

                            flash(
                                f"Capped {capped} outlier value(s) "
                                f"in `{out_col}` using {method} bounds."
                            )

                            st.rerun()

                    except Exception as e:
                        drop_last_history()

                        self.df = st.session_state.df.copy()

                        st.error(
                            f"Outlier capping failed: {e}"
                        )

            else:
                st.info(
                    "No numerical columns present for outlier processing."
                )

        self._render_quality_panel()

    def _render_quality_panel(self):
        st.divider()

        st.subheader("Data Quality Improvement Score")

        orig_df = st.session_state.get(
            "original_df",
            None
        )

        if (
            orig_df is None
            or orig_df.size == 0
            or self.df.size == 0
        ):
            st.info(
                "Load a dataset to see quality metrics."
            )
            return

        orig_missing = int(
            orig_df.isnull().sum().sum()
        )

        clean_missing = int(
            self.df.isnull().sum().sum()
        )

        missing_resolved = (
            orig_missing - clean_missing
        )

        missing_pct_cleaned = (
            missing_resolved / orig_missing * 100
            if orig_missing > 0
            else 100.0
        )

        orig_dups = safe_duplicate_count(orig_df)

        clean_dups = safe_duplicate_count(self.df)

        dups_resolved = (
            orig_dups - clean_dups
        )

        dups_pct_cleaned = (
            dups_resolved / orig_dups * 100
            if orig_dups > 0
            else 100.0
        )

        def health(frame, missing, dups):
            cells = frame.shape[0] * frame.shape[1]

            issues = (
                missing
                + dups * frame.shape[1]
            )

            return max(
                0.0,
                min(
                    100.0,
                    (cells - issues) / cells * 100
                )
            )

        orig_health = health(
            orig_df,
            orig_missing,
            orig_dups
        )

        clean_health = health(
            self.df,
            clean_missing,
            clean_dups
        )

        improvement = (
            clean_health - orig_health
        )

        m1, m2, m3 = st.columns(3)

        m1.metric(
            "Missing Values Resolved",
            f"{missing_pct_cleaned:.1f}%",
            f"{missing_resolved} fixed"
        )

        m2.metric(
            "Duplicates Removed",
            f"{dups_pct_cleaned:.1f}%",
            f"{dups_resolved} fixed"
        )

        m3.metric(
            "Overall Data Health",
            f"{clean_health:.1f}%",
            f"{improvement:+.1f}% change"
        )

        st.write(
            f"**Data Cleaned Readiness Progress:** "
            f"`{clean_health:.1f}%`"
        )

        st.progress(
            min(
                1.0,
                max(
                    0.0,
                    clean_health / 100.0
                )
            )
        )

        st.caption(
            f"Fixed **{missing_resolved} missing values** "
            f"and **{dups_resolved} duplicate rows** so far."
        )
