import io
import time

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import (
    AdaBoostClassifier, AdaBoostRegressor, ExtraTreesClassifier, ExtraTreesRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, mean_absolute_error, mean_squared_error,
    precision_score, r2_score, recall_score,
)
from sklearn.model_selection import ParameterGrid, RandomizedSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from pipeline.theme import chart_template, get_theme
from pipeline.utils import (
    clear_ml_state, column_groups, ui_button, ui_dataframe, ui_download, ui_plot,
)

MAX_CATEGORIES = 50          # categorical columns with more levels are dropped from the features
FAST_ROW_CAP = 10_000
DEEP_ROW_CAP = 30_000
SVM_ROW_CAP = 5_000
MODEL_CLASS_LIMIT_WARN = 50  # classification targets with more classes get a warning

CLASSIFIERS = {
    "Logistic Regression": lambda: LogisticRegression(max_iter=1000),
    "Decision Tree": lambda: DecisionTreeClassifier(random_state=42),
    "Random Forest": lambda: RandomForestClassifier(n_jobs=-1, n_estimators=50, random_state=42),
    "Gradient Boosting": lambda: GradientBoostingClassifier(n_estimators=50, random_state=42),
    "AdaBoost": lambda: AdaBoostClassifier(n_estimators=30, random_state=42),
    "Extra Trees": lambda: ExtraTreesClassifier(n_jobs=-1, n_estimators=50, random_state=42),
    "K-Nearest Neighbors": lambda: KNeighborsClassifier(n_jobs=-1),
    "Naive Bayes": lambda: GaussianNB(),
    "Support Vector Machine (SVC)": lambda: SVC(),
}
REGRESSORS = {
    "Linear Regression": lambda: LinearRegression(),
    "Ridge Regression": lambda: Ridge(),
    "Lasso Regression": lambda: Lasso(),
    "ElasticNet": lambda: ElasticNet(),
    "Decision Tree": lambda: DecisionTreeRegressor(random_state=42),
    "Random Forest": lambda: RandomForestRegressor(n_jobs=-1, n_estimators=50, random_state=42),
    "Gradient Boosting": lambda: GradientBoostingRegressor(n_estimators=50, random_state=42),
    "AdaBoost": lambda: AdaBoostRegressor(n_estimators=30, random_state=42),
    "Extra Trees": lambda: ExtraTreesRegressor(n_jobs=-1, n_estimators=50, random_state=42),
    "K-Nearest Neighbors": lambda: KNeighborsRegressor(n_jobs=-1),
    "Support Vector Regressor (SVR)": lambda: SVR(),
}

_TREE_GRID = {"max_depth": [3, 5, 10, None], "min_samples_split": [2, 10]}
_FOREST_GRID = {"n_estimators": [50, 100], "max_depth": [5, 10, None]}
_BOOST_GRID = {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1, 0.2], "max_depth": [2, 3]}
_ADA_GRID = {"n_estimators": [30, 50, 100], "learning_rate": [0.5, 1.0]}
_KNN_GRID = {"n_neighbors": [3, 5, 9, 15], "weights": ["uniform", "distance"]}
PARAM_GRIDS = {
    "Classification": {
        "Logistic Regression": {"C": [0.1, 1, 10]},
        "Decision Tree": _TREE_GRID, "Random Forest": _FOREST_GRID, "Extra Trees": _FOREST_GRID,
        "Gradient Boosting": _BOOST_GRID, "AdaBoost": _ADA_GRID, "K-Nearest Neighbors": _KNN_GRID,
        "Naive Bayes": {"var_smoothing": [1e-9, 1e-8, 1e-7]},
        "Support Vector Machine (SVC)": {"C": [0.5, 1, 5]},
    },
    "Regression": {
        "Ridge Regression": {"alpha": [0.1, 1, 10, 100]},
        "Lasso Regression": {"alpha": [0.001, 0.01, 0.1, 1]},
        "ElasticNet": {"alpha": [0.01, 0.1, 1], "l1_ratio": [0.2, 0.5, 0.8]},
        "Decision Tree": _TREE_GRID, "Random Forest": _FOREST_GRID, "Extra Trees": _FOREST_GRID,
        "Gradient Boosting": _BOOST_GRID, "AdaBoost": _ADA_GRID, "K-Nearest Neighbors": _KNN_GRID,
        "Support Vector Regressor (SVR)": {"C": [0.5, 1, 5]},
    },
}


class AutoMLEngine:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    # ------------------------------------------------------------------ preprocessing
    def preprocess_data(self, target_col: str, task_type: str):
        """Clean, encode, split and scale. Scaling is fitted on the TRAIN split only (no leakage).
        Returns a dict, or None after showing an error."""
        df = self.df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=[target_col])
        if len(df) < 10:
            st.error("❌ Need at least 10 rows with a non-missing target.")
            return None

        label_encoder, class_names = None, None
        y_raw = df[target_col]

        if task_type == "Regression":
            y_num = pd.to_numeric(y_raw, errors="coerce")
            keep = y_num.notna()
            df, y = df[keep], y_num[keep].astype("float64").to_numpy()
            if len(y) < 10:
                st.error(f"❌ Target column **'{target_col}'** has too few numeric values for regression.")
                return None
        else:
            if y_raw.nunique() < 2:
                st.error("❌ The target has only one class — classification needs at least 2.")
                return None
            label_encoder = LabelEncoder()
            try:
                y = label_encoder.fit_transform(y_raw)
            except TypeError:
                y = label_encoder.fit_transform(y_raw.astype(str))
            class_names = [str(c) for c in label_encoder.classes_]
            if len(class_names) > MODEL_CLASS_LIMIT_WARN:
                st.warning(f"⚠️ Target has {len(class_names)} classes — did you mean Regression?")

        X = df.drop(columns=[target_col])
        notes = []

        # Drop columns that cannot help or would explode the feature space
        num_cols, cat_cols, dt_cols = column_groups(X)
        drop = {}
        for c in dt_cols:
            drop[c] = "datetime column (extract date features first)"
        for c in X.columns:
            if c in drop:
                continue
            if X[c].isna().all():
                drop[c] = "only missing values"
                continue
            try:
                n_unique = X[c].nunique(dropna=True)
            except TypeError:
                drop[c] = "contains unhashable values (lists/dicts)"
                continue
            if n_unique <= 1:
                drop[c] = "constant column"
            elif c in cat_cols and n_unique > MAX_CATEGORIES:
                drop[c] = f"too many categories ({n_unique} > {MAX_CATEGORIES}) — likely an ID/free text"
        if drop:
            X = X.drop(columns=list(drop))
            notes.extend(f"`{c}` dropped: {why}" for c, why in drop.items())
        cat_cols = [c for c in cat_cols if c in X.columns]
        if X.shape[1] == 0:
            st.error("❌ No usable feature columns remain after cleaning.")
            return None

        input_columns = X.columns.tolist()
        X = pd.get_dummies(X, columns=cat_cols, dtype=float)
        X = X.apply(pd.to_numeric, errors="coerce").astype("float64")
        X = X.replace([np.inf, -np.inf], np.nan)

        # Split first, then impute/scale using training statistics only
        stratify = None
        if task_type == "Classification":
            counts = np.bincount(y)
            if counts.min() >= 2 and len(counts) <= int(np.ceil(len(y) * 0.2)):
                stratify = y
            else:
                notes.append("Stratified split disabled (a class has <2 rows or there are too many classes).")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify)

        fill_values = X_train.median(numeric_only=True).fillna(0)
        X_train = X_train.fillna(fill_values).fillna(0)
        X_test = X_test.fillna(fill_values).fillna(0)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        return {
            "X_train": X_train_s, "X_test": X_test_s,
            "y_train": np.asarray(y_train), "y_test": np.asarray(y_test),
            "feature_names": X.columns.tolist(), "scaler": scaler, "label_encoder": label_encoder,
            "fill_values": fill_values, "class_names": class_names,
            "input_columns": input_columns, "cat_cols": cat_cols, "notes": notes,
        }

    # ------------------------------------------------------------------ training
    @staticmethod
    def _fit_one(name, factory, task_type, deep, cv_folds, X_tr, y_tr):
        model = factory()
        grid = PARAM_GRIDS[task_type].get(name)
        if deep and grid:
            cv = cv_folds
            if task_type == "Classification":
                cv = min(cv_folds, int(np.bincount(y_tr).min()))
            if cv >= 2:
                n_iter = min(8, len(ParameterGrid(grid)))
                try:
                    searcher = RandomizedSearchCV(model, grid, n_iter=n_iter, cv=cv, n_jobs=-1, random_state=42)
                    searcher.fit(X_tr, y_tr)
                    return searcher.best_estimator_, f" (best params: {searcher.best_params_})"
                except Exception:
                    model = factory()  # fall back to a plain fit below
        model.fit(X_tr, y_tr)
        return model, ""

    @staticmethod
    def _score(task_type, y_true, preds, name, seconds):
        if task_type == "Classification":
            return {
                "Model": name,
                "Accuracy": round(accuracy_score(y_true, preds), 4),
                "Precision": round(precision_score(y_true, preds, average="weighted", zero_division=0), 4),
                "Recall": round(recall_score(y_true, preds, average="weighted", zero_division=0), 4),
                "F1-Score": round(f1_score(y_true, preds, average="weighted", zero_division=0), 4),
                "Time (s)": round(seconds, 2),
            }
        return {
            "Model": name,
            "RMSE": round(float(np.sqrt(mean_squared_error(y_true, preds))), 4),
            "MAE": round(mean_absolute_error(y_true, preds), 4),
            "R² Score": round(r2_score(y_true, preds), 4),
            "Time (s)": round(seconds, 2),
        }

    # ------------------------------------------------------------------ UI
    def render_studio(self):
        st.title("🤖 Enterprise AutoML & Fast Benchmarking Studio")

        if self.df is None or self.df.empty:
            st.error("⚠️ Please Upload Dataset")
            return

        # Drop stale results when the dataset changed
        if st.session_state.get("ml_source_file") not in (None, st.session_state.get("file_name")):
            clear_ml_state()

        all_cols = self.df.columns.tolist()
        st.subheader("🎯 Choose Target Column")
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            target_col = st.selectbox("Select Target Variable (Y)", all_cols, key="ml_target")

        try:
            n_unique = self.df[target_col].nunique()
        except TypeError:
            n_unique = 10 ** 9
        is_numeric_target = pd.api.types.is_numeric_dtype(self.df[target_col]) and \
            not pd.api.types.is_bool_dtype(self.df[target_col])
        default_task = "Regression" if (is_numeric_target and n_unique >= 15) else "Classification"
        with col_t2:
            task_type = st.radio("ML Problem Type", ["Classification", "Regression"],
                                 index=0 if default_task == "Classification" else 1, horizontal=True,
                                 key=f"ml_task_{target_col}")

        st.divider()
        st.subheader("⚡ Choose Model")
        factories = CLASSIFIERS if task_type == "Classification" else REGRESSORS
        selected = st.multiselect(
            "Select multiple models for comparison", options=list(factories.keys()),
            default=list(factories.keys())[:3], key=f"ml_models_{task_type}")

        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            speed_mode = st.radio("⚡ Execution Speed Mode",
                                  ["Fast Mode (Quick Benchmarking)", "Deep Search (Hyperparameter Tuning)"],
                                  index=0, key="ml_speed")
        deep = speed_mode.startswith("Deep")
        cv_folds = 3
        with col_opt2:
            if deep:
                cv_folds = st.slider("CV Folds", 3, 5, 3, key="ml_cv")

        st.divider()

        if ui_button("🚀 Run AutoML Pipeline", key="ml_run"):
            if not selected:
                st.warning("⚠️ Select at least one model.")
            else:
                self._run_pipeline(target_col, task_type, selected, factories, deep, cv_folds)

        if "ml_results" in st.session_state:
            self._render_results()
            st.divider()
            self._render_prediction_tab()

    def _run_pipeline(self, target_col, task_type, selected, factories, deep, cv_folds):
        prep = self.preprocess_data(target_col, task_type)
        if prep is None:
            return
        X_train, X_test, y_train, y_test = prep["X_train"], prep["X_test"], prep["y_train"], prep["y_test"]

        rng = np.random.default_rng(42)
        cap = DEEP_ROW_CAP if deep else FAST_ROW_CAP

        def subsample(X, y, limit):
            if len(X) <= limit:
                return X, y
            idx = rng.choice(len(X), limit, replace=False)
            return X[idx], y[idx]

        X_tr, y_tr = subsample(X_train, y_train, cap)
        if len(X_tr) < len(X_train):
            prep["notes"].append(f"Training used a {len(X_tr):,}-row sample for speed.")

        results, trained, failures = [], {}, []
        progress = st.progress(0.0)
        status = st.empty()
        start = time.time()

        for i, name in enumerate(selected):
            progress.progress(i / len(selected))
            status.markdown(f"⏳ **Training ({i + 1}/{len(selected)}):** `{name}`...")
            t0 = time.time()
            try:
                X_fit, y_fit = (subsample(X_tr, y_tr, SVM_ROW_CAP) if "SV" in name else (X_tr, y_tr))
                if task_type == "Classification" and len(np.unique(y_fit)) < 2:
                    raise ValueError("training sample contains a single class")
                model, extra = self._fit_one(name, factories[name], task_type, deep, cv_folds, X_fit, y_fit)
                preds = model.predict(X_test)
                trained[name] = model
                results.append(self._score(task_type, y_test, preds, name, time.time() - t0))
                if extra:
                    prep["notes"].append(f"{name}{extra}")
            except Exception as e:
                failures.append(f"`{name}` failed: {e}")

        progress.progress(1.0)
        elapsed = round(time.time() - start, 2)
        status.empty()

        if not results:
            st.error("❌ Every selected model failed to train.\n\n" + "\n".join(f"- {f}" for f in failures))
            return

        sort_col = "Accuracy" if task_type == "Classification" else "R² Score"
        res_df = pd.DataFrame(results).sort_values(by=sort_col, ascending=False).reset_index(drop=True)
        best_name = res_df.iloc[0]["Model"]
        best_model = trained[best_name]

        buffer = io.BytesIO()
        joblib.dump({
            "model": best_model, "model_name": best_name, "scaler": prep["scaler"],
            "label_encoder": prep["label_encoder"], "feature_names": prep["feature_names"],
            "fill_values": prep["fill_values"], "input_columns": prep["input_columns"],
            "cat_cols": prep["cat_cols"], "task_type": task_type, "target": target_col,
        }, buffer)

        ss = st.session_state
        ss.ml_results = res_df
        ss.ml_best_name = best_name
        ss.ml_best_model = best_model
        ss.ml_feature_names = prep["feature_names"]
        ss.ml_scaler = prep["scaler"]
        ss.ml_label_encoder = prep["label_encoder"]
        ss.ml_class_names = prep["class_names"]
        ss.ml_fill_values = prep["fill_values"]
        ss.ml_input_columns = prep["input_columns"]
        ss.ml_cat_cols = prep["cat_cols"]
        ss.ml_task_type = task_type
        ss.ml_target_col = target_col
        ss.ml_test_data = (X_test, y_test)
        ss.ml_trained_models = trained
        ss.ml_sort_col = sort_col
        ss.ml_elapsed = elapsed
        ss.ml_model_bytes = buffer.getvalue()
        ss.ml_source_file = ss.get("file_name")
        ss.ml_notes = prep["notes"] + failures

    def _render_results(self):
        ss = st.session_state
        res_df, sort_col, task_type = ss.ml_results, ss.ml_sort_col, ss.ml_task_type
        theme = get_theme()

        st.success(f"✅ Training completed in **{ss.ml_elapsed} seconds**!")
        for note in ss.get("ml_notes", []):
            st.caption(f"ℹ️ {note}")

        st.subheader("🏆 Leaderboard Benchmark Results")
        ui_dataframe(res_df)

        fig = px.bar(res_df, x="Model", y=sort_col, color=sort_col, text_auto=True,
                     color_continuous_scale=theme["scale"], title=f"📊 Performance Comparison ({sort_col})",
                     template=chart_template())
        ui_plot(fig, key="ml_comp_fig")

        best = res_df.iloc[0]
        st.success(f"🥇 **Best Model:** `{ss.ml_best_name}` ({sort_col}: **{best[sort_col]}**)")

        self._render_diagnostics(ss.ml_best_name, ss.ml_best_model, *ss.ml_test_data, task_type)

        ui_download(
            f"📥 Download Best Model ({ss.ml_best_name}) as .joblib",
            data=ss.ml_model_bytes,
            file_name=f"{ss.ml_best_name.replace(' ', '_').replace('(', '').replace(')', '').lower()}_model.joblib",
            mime="application/octet-stream", key="ml_dl_model",
        )
        st.caption("The file bundles the model with its scaler, encoders and feature list "
                   "(`joblib.load(...)` returns a dict).")

    @staticmethod
    def _importances(model, n_features):
        if hasattr(model, "feature_importances_"):
            vals = np.asarray(model.feature_importances_)
        elif hasattr(model, "coef_"):
            vals = np.abs(np.asarray(model.coef_))
            vals = vals.mean(axis=0) if vals.ndim > 1 else vals
        else:
            return None
        return vals if vals.shape[0] == n_features else None

    def _render_diagnostics(self, name, model, X_test, y_test, task_type):
        st.divider()
        st.subheader(f"🔬 Diagnostics — {name}")
        theme, template = get_theme(), chart_template()
        feat_names = st.session_state.ml_feature_names
        c1, c2 = st.columns(2)

        with c1:
            vals = self._importances(model, len(feat_names))
            if vals is not None:
                imp = pd.DataFrame({"Feature": feat_names, "Importance": vals}) \
                    .sort_values("Importance", ascending=False).head(15).iloc[::-1]
                fig = px.bar(imp, x="Importance", y="Feature", orientation="h", title="Top 15 Feature Importances",
                             color="Importance", color_continuous_scale=theme["scale"], template=template)
                ui_plot(fig, key="ml_imp_fig")
            else:
                st.info("Feature importance isn't available for this model type.")

        with c2:
            preds = model.predict(X_test)
            if task_type == "Classification":
                names = st.session_state.ml_class_names or []
                if len(names) > 30:
                    st.info("Too many classes to draw a readable confusion matrix.")
                else:
                    cm = confusion_matrix(y_test, preds, labels=np.arange(len(names)))
                    fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale=theme["scale"],
                                       x=names, y=names, title="Confusion Matrix",
                                       labels=dict(x="Predicted", y="Actual"), template=template)
                    ui_plot(fig_cm, key="ml_cm_fig")
            else:
                residuals = y_test - preds
                fig_res = px.scatter(x=preds, y=residuals, title="Residual Plot",
                                     labels={"x": "Predicted", "y": "Residual"}, template=template,
                                     color_discrete_sequence=[theme["accent"]])
                fig_res.add_hline(y=0, line_dash="dash", line_color=theme["accent2"])
                ui_plot(fig_res, key="ml_res_fig")

        if task_type == "Regression":
            lo, hi = float(min(y_test.min(), preds.min())), float(max(y_test.max(), preds.max()))
            fig_ap = px.scatter(x=y_test, y=preds, title="Actual vs Predicted",
                                labels={"x": "Actual", "y": "Predicted"}, template=template,
                                color_discrete_sequence=[theme["accent"]])
            fig_ap.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi,
                             line=dict(dash="dash", color=theme["accent2"]))
            ui_plot(fig_ap, key="ml_ap_fig")

    def _render_prediction_tab(self):
        ss = st.session_state
        st.subheader("🔮 Predict on New Data")
        st.caption("Upload a CSV with the same feature columns (the target column is optional and ignored).")
        new_file = st.file_uploader("Upload CSV for Prediction", type=["csv"], key="ml_predict_upload")
        if new_file is None:
            return

        try:
            try:
                new_df = pd.read_csv(new_file, encoding="utf-8-sig")
            except UnicodeDecodeError:
                new_file.seek(0)
                new_df = pd.read_csv(new_file, encoding="latin-1")
            new_df.columns = [str(c).strip() for c in new_df.columns]

            missing = [c for c in ss.ml_input_columns if c not in new_df.columns]
            if missing:
                st.error("❌ Missing required column(s): " + ", ".join(f"`{c}`" for c in missing))
                return

            X_new = new_df[ss.ml_input_columns].replace([np.inf, -np.inf], np.nan)
            X_new = pd.get_dummies(X_new, columns=[c for c in ss.ml_cat_cols if c in X_new.columns], dtype=float)
            X_new = X_new.apply(pd.to_numeric, errors="coerce")
            X_new = X_new.reindex(columns=ss.ml_feature_names, fill_value=0).astype("float64")
            X_new = X_new.fillna(ss.ml_fill_values).fillna(0)
            preds = ss.ml_best_model.predict(ss.ml_scaler.transform(X_new))

            if ss.ml_task_type == "Classification" and ss.ml_label_encoder is not None:
                preds = ss.ml_label_encoder.inverse_transform(np.asarray(preds).astype(int))

            result_df = new_df.copy()
            result_df[f"Predicted_{ss.ml_target_col}"] = preds
            ui_dataframe(result_df)
            ui_download("📥 Download Predictions (CSV)", data=result_df.to_csv(index=False).encode("utf-8"),
                        file_name="predictions.csv", mime="text/csv", key="ml_dl_preds")
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")

    # Alias for compatibility with app.py
    render_ml_interface = render_studio
