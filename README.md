# ⚡ AutoDataLab Enterprise

An all-in-one **Streamlit** app that takes a dataset from upload to trained model:

**Upload → Explore → Clean → Engineer → Visualize → Model → Ask**

Dark **Sunset Ember** theme (deep brown background, orange and rose accents) with hover effects on buttons, cards and tabs.

---

## ✨ Features

| Step | What you can do |
|------|-----------------|
| 📂 **Data Loader** | Upload CSV, Excel (with sheet picker), JSON or Parquet. Handles encodings, `;` and tab delimiters, empty files and duplicate column names. Undo, reset and CSV download. |
| 🔍 **Data Explorer** | Row, column, duplicate and missing-value metrics. Dataset preview, column summary, numeric and categorical statistics, correlation heatmap, per-column distribution. |
| 🧹 **Data Cleaner** | Missing values (mean, median, mode, constant, ffill, bfill, drop). Duplicate removal. Data type conversion. Outlier capping (IQR and Z-score). Live data-quality score. |
| ⚙️ **Feature Engineering** | Scaling (Standard, MinMax, Robust). Label and one-hot encoding. Polynomial features. Date feature extraction. Binning. Custom formulas such as `income / (age + 1)`. |
| 📊 **Dashboard (EDA)** | Category and numeric slicers, KPI cards, and univariate, bivariate and multivariate charts. Outlier and skewness analysis, missing-value matrix, and a custom chart builder with single and dual canvas. |
| 🤖 **ML Studio** | Classification and regression. Up to 9 or 11 models side by side, with a fast mode or a hyperparameter-search mode. Leaderboard, feature importance, confusion matrix or residual plots, predictions on a new CSV, and model download. |
| 💬 **AI Data Chatbot** | Loaded from your own `pipeline/chatbot.py` (see below). |

### 🎨 EDA Color Studio
Open **Chart Color Studio** on the Dashboard page to pick:
- **Category palette**: app theme, 10 built-in palettes (including a color-blind-safe one), or 5 custom color pickers.
- **Continuous color scale**: 17 options, with a reverse toggle.
- **Chart style**: auto (matches the app) or any Plotly template.
- **Single color** for any custom chart.

---

## 📁 Project Structure

```
your_project/
├── app.py
├── smoke_test.py
└── pipeline/
    ├── __init__.py
    ├── data_loader.py
    ├── data_explorer.py
    ├── cleaner.py
    ├── feature_engineering.py
    ├── eda_dashboard.py
    ├── ml_tuner.py
    ├── theme.py
    ├── utils.py
    └── chatbot.py        ← your own file (not included in the update zip)
```

---

## 🚀 Getting Started

**1. Install dependencies** (Python 3.9+ recommended)

```bash
pip install streamlit pandas numpy plotly scikit-learn statsmodels joblib pyarrow openpyxl xlrd
```

> `statsmodels` is only needed for the OLS trendline option. `openpyxl` and `xlrd` are only needed for Excel files. Your chatbot may need extra packages.

**2. Run the app**

```bash
streamlit run app.py
```

**3. Open** the local URL Streamlit prints (usually http://localhost:8501).

---

## 🧭 How to Use

1. Go to **Data Loader** and upload a file.
2. Use the sidebar to move between steps. Every cleaning and engineering change can be undone from **Data Loader → ↩️ Undo Last Change**, or fully reset with **🔄 Reset to Original**.
3. On the **ML Studio** page, pick a target column, choose the problem type (guessed automatically, and you can change it), select models, and press **🚀 Run AutoML Pipeline**.
4. Download the best model as a `.joblib` file. It is a dictionary containing the model, scaler, encoders and feature list:

```python
import joblib
bundle = joblib.load("random_forest_model.joblib")
model, scaler = bundle["model"], bundle["scaler"]
```

---

## 🛡️ Built-in Safeguards

- Training never leaks test data into the scaler.
- Columns that cannot help a model (IDs, free text, constant columns, dates) are skipped, and the app tells you which ones.
- One failing model does not stop the rest.
- One-hot encoding and polynomial features have size limits to protect memory.
- Large datasets are sampled for plots and training speed.
- Errors appear on the page instead of crashing the app.
- Custom formulas use `df.eval`, with `@` and `__` blocked.

---

## 🧪 Testing

```bash
python smoke_test.py
```

This runs the app headlessly with sample data, visits each page and runs a classification and a regression training pass. It needs `streamlit`, `plotly` and a `pipeline/chatbot.py` file to exist.

---

## ⚠️ Known Limitations

- Streamlit reruns the page after each action, so the Cleaner and Feature Engineering pages use a section selector instead of tabs to keep your place.
- Training uses a sample of at most 10,000 rows in fast mode (30,000 in deep search), and 5,000 rows for SVM models.
- The chatbot depends entirely on your own `chatbot.py`.

---

## 📄 License

Add your preferred license here.
