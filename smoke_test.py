import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

rng = np.random.default_rng(0)
n = 300
df = pd.DataFrame({
    "age": rng.integers(18, 80, n).astype(float),
    "income": rng.normal(50000, 15000, n),
    "city": rng.choice(["Delhi", "Pune", "Noida", None], n),
    "joined": pd.date_range("2022-01-01", periods=n, freq="D").astype(str),
    "score": rng.normal(0, 1, n),
    "label": rng.choice(["yes", "no"], n),
})
df.loc[rng.choice(n, 20, replace=False), "age"] = np.nan
df = pd.concat([df, df.head(5)], ignore_index=True)

at = AppTest.from_file("app.py", default_timeout=120)
at.session_state["df"] = df.copy()
at.session_state["original_df"] = df.copy()
at.session_state["file_name"] = "test.csv"
at.session_state["file_sig"] = ("test.csv", 1, None)
at.session_state["history"] = []
at.run()
assert not at.exception, at.exception

from pipeline.utils import ML_STATE_KEYS  # noqa
PAGES = [o for o in at.sidebar.radio[0].options]
for page in PAGES[:6]:
    at.sidebar.radio[0].set_value(page).run()
    print(page, "-> exceptions:", [e.value for e in at.exception])
    assert not at.exception, at.exception

for theme in at.sidebar.selectbox[0].options:
    at.sidebar.selectbox[0].set_value(theme).run()
    assert not at.exception, at.exception
print("themes OK")

# ML page: click run
at.sidebar.radio[0].set_value(PAGES[5]).run()
for b in at.button:
    if b.key == "ml_run":
        b.click()
at.run()
print("ML exceptions:", [e.value for e in at.exception], "errors:", [e.value for e in at.error])
assert not at.exception
assert "ml_results" in at.session_state

# ML regression
at.selectbox(key="ml_target").set_value("income").run()
print("target income task:", at.radio(key="ml_task_income").value)
for b in at.button:
    if b.key == "ml_run":
        b.click()
at.run()
print("Regression exceptions:", [e.value for e in at.exception], "errors:", [e.value for e in at.error])
assert not at.exception
print(at.session_state["ml_results"])

# Cleaner action
at.sidebar.radio[0].set_value(PAGES[2]).run()
for b in at.button:
    if b.key == "btn_missing":
        b.click()
at.run()
print("clean exceptions:", [e.value for e in at.exception], "errors:", [e.value for e in at.error])
assert not at.exception
print("SMOKE TEST PASSED")
