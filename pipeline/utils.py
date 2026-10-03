"""Shared helpers for AutoDataLab (history, flash messages, safe widgets, safe pandas helpers)."""
import streamlit as st
import pandas as pd

# Session-state keys produced by the ML studio (cleared when a new dataset is loaded)
ML_STATE_KEYS = [
    "ml_best_model", "ml_feature_names", "ml_scaler", "ml_label_encoder", "ml_task_type",
    "ml_target_col", "ml_test_data", "ml_trained_models", "ml_results", "ml_best_name",
    "ml_fill_values", "ml_source_file", "ml_class_names", "ml_elapsed", "ml_sort_col",
    "ml_input_columns", "ml_cat_cols", "ml_model_bytes", "ml_notes",
]


def clear_ml_state():
    for key in ML_STATE_KEYS:
        st.session_state.pop(key, None)


# ---------------------------------------------------------------------------
# Streamlit-version-safe "stretch to container width" wrappers.
# New Streamlit uses width="stretch", older uses use_container_width=True.
# ---------------------------------------------------------------------------
_STRETCH_MODE = {}


def _stretch_call(fn, *args, **kwargs):
    name = getattr(fn, "__name__", str(fn))
    mode = _STRETCH_MODE.get(name)
    if mode != "old":
        try:
            result = fn(*args, width="stretch", **kwargs)
            _STRETCH_MODE[name] = "new"
            return result
        except Exception as first_err:
            if mode == "new":
                raise
            try:
                result = fn(*args, use_container_width=True, **kwargs)
            except Exception:
                raise first_err
            _STRETCH_MODE[name] = "old"
            return result
    return fn(*args, use_container_width=True, **kwargs)


def ui_dataframe(df, **kwargs):
    return _stretch_call(st.dataframe, arrow_safe(df), **kwargs)


def ui_plot(fig, key=None, **kwargs):
    if st.session_state.get("_transparent_charts", True):
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    if key is not None:
        kwargs["key"] = key
    return _stretch_call(st.plotly_chart, fig, **kwargs)


def ui_button(label, **kwargs):
    return _stretch_call(st.button, label, **kwargs)


def ui_download(label, **kwargs):
    return _stretch_call(st.download_button, label, **kwargs)


# ---------------------------------------------------------------------------
# Flash messages (survive st.rerun(), unlike st.success called right before it)
# ---------------------------------------------------------------------------
def flash(message: str, kind: str = "success"):
    st.session_state["_flash"] = (kind, message)


def show_flash():
    item = st.session_state.pop("_flash", None)
    if item:
        kind, message = item
        getattr(st, kind, st.info)(message)


# ---------------------------------------------------------------------------
# Undo history
# ---------------------------------------------------------------------------
def push_history(df: pd.DataFrame):
    hist = st.session_state.setdefault("history", [])
    hist.append(df.copy())
    try:
        mem_mb = df.memory_usage(deep=False).sum() / 1e6
    except Exception:
        mem_mb = 0
    limit = 15 if mem_mb < 50 else 5
    while len(hist) > limit:
        hist.pop(0)


def drop_last_history():
    hist = st.session_state.get("history", [])
    if hist:
        hist.pop()


# ---------------------------------------------------------------------------
# DataFrame helpers
# ---------------------------------------------------------------------------
def column_groups(df: pd.DataFrame):
    """Return (numeric_cols, categorical_cols, datetime_cols). Bool counts as categorical."""
    num, cat, dt = [], [], []
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_bool_dtype(s):
            cat.append(col)
        elif pd.api.types.is_numeric_dtype(s):
            num.append(col)
        elif pd.api.types.is_datetime64_any_dtype(s):
            dt.append(col)
        else:
            cat.append(col)
    return num, cat, dt


def arrow_safe(df):
    """Convert mixed-type object columns to str so st.dataframe (Arrow) never crashes."""
    if not isinstance(df, pd.DataFrame):
        return df
    out = df
    copied = False
    for i in range(df.shape[1]):
        s = df.iloc[:, i]
        if s.dtype == object:
            try:
                kind = pd.api.types.infer_dtype(s, skipna=True)
            except Exception:
                kind = "mixed"
            if kind.startswith("mixed") or kind == "unknown-array":
                if not copied:
                    out = df.copy()
                    copied = True
                out.isetitem(i, s.astype(str))
    return out


def safe_duplicated(df: pd.DataFrame) -> pd.Series:
    try:
        return df.duplicated()
    except TypeError:  # unhashable cells such as lists / dicts
        return df.astype(str).duplicated()


def safe_duplicate_count(df: pd.DataFrame) -> int:
    return int(safe_duplicated(df).sum())
