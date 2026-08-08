"""
Wine Quality Classifier - Streamlit app
=======================================
Interactive front-end for five classification models trained on the UCI Wine
Quality dataset (red + white combined). The models predict whether a wine is
"good" (quality >= 7) from its 12 physicochemical features.

The models are trained ON STARTUP from the UCI data (cached for the session) --
no pre-pickled model files are shipped, which keeps the repo lightweight and
avoids scikit-learn version-mismatch errors on Streamlit Cloud. Training all five
models takes under a second.

Required features implemented (per assignment Step 6):
  a. CSV upload (test data)
  b. Model selection dropdown
  c. Evaluation-metrics display (Accuracy, AUC, Precision, Recall, F1, MCC)
  d. Confusion matrix + classification report

Run locally:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report,
)

st.set_page_config(page_title="Wine Quality Classifier", page_icon="🍷", layout="wide")

RED_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
WHITE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
TARGET = "good"
RANDOM_STATE = 42


def build_models():
    """The five classification models required by the assignment."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE),
        "kNN": KNeighborsClassifier(n_neighbors=15),
        "Naive Bayes": GaussianNB(),
        "Random Forest (Ensemble)": RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
        ),
    }


@st.cache_resource(show_spinner="Training models on UCI Wine Quality data…")
def load_scaler_and_models():
    """Download the UCI data, build the target, fit the scaler and all 5 models.

    Cached for the whole session, so this runs only once. Returns the fitted
    scaler, the feature-column list and the dict of fitted models.
    """
    red = pd.read_csv(RED_URL, sep=";")
    white = pd.read_csv(WHITE_URL, sep=";")
    red["wine_type"] = 0
    white["wine_type"] = 1
    df = pd.concat([red, white], ignore_index=True)
    df[TARGET] = (df["quality"] >= 7).astype(int)
    df = df.drop(columns=["quality"])

    feature_cols = [c for c in df.columns if c != TARGET]
    X = df[feature_cols].values
    y = df[TARGET].values

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)

    models = build_models()
    for m in models.values():
        m.fit(X_train_s, y_train)
    return scaler, feature_cols, models


def compute_metrics(y_true, y_pred, y_score):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_score),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def scores_for(model, X):
    return model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X)


# ----------------------------------------------------------------------------- Styling
st.markdown(
    """
    <style>
      /* Constrain overall content width and centre it */
      .block-container { max-width: 1050px; padding-top: 2rem; }

      /* Wine-themed hero banner */
      .hero {
        background: linear-gradient(135deg, #6d213c 0%, #a8324a 55%, #c85a6e 100%);
        border-radius: 16px; padding: 26px 32px; margin-bottom: 22px;
        color: #fff; box-shadow: 0 6px 22px rgba(109,33,60,.28);
      }
      .hero h1 { margin: 0; font-size: 2.05rem; font-weight: 800; letter-spacing:-.5px; }
      .hero p  { margin: 8px 0 0; font-size: 1.02rem; opacity: .93; max-width: 760px; }

      /* Section labels */
      .step { font-size: 1.15rem; font-weight: 700; color: #6d213c;
              margin: 6px 0 2px; }

      /* Metric tiles */
      div[data-testid="stMetric"] {
        background: #faf3f5; border: 1px solid #ecd7dd; border-radius: 12px;
        padding: 12px 14px; text-align: center;
      }
      div[data-testid="stMetricLabel"] { justify-content:center; color:#8a4a5c; font-weight:600; }
      div[data-testid="stMetricValue"] { font-size: 1.6rem; color:#6d213c; }

      /* Tighten the file-uploader box */
      section[data-testid="stFileUploaderDropzone"] { padding: 10px 14px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------- UI
st.markdown(
    """
    <div class="hero">
      <h1>🍷 Wine Quality Classifier</h1>
      <p>Predict whether a wine is <b>good</b> (quality ≥ 7) from its 12 physicochemical
      properties, and compare five classification models. Pick a model and upload your
      test CSV to evaluate it.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

scaler, feature_cols, models = load_scaler_and_models()

# ---- Controls in the MAIN area (a: CSV upload, b: model dropdown) ----
st.markdown('<div class="step">1 · Choose a model &nbsp;&amp;&nbsp; upload your test data</div>',
            unsafe_allow_html=True)

# Narrow the controls so they don't stretch edge-to-edge: model | upload | spacer.
c_model, c_upload, _spacer = st.columns([1.1, 1.4, 0.5])
with c_model:
    model_name = st.selectbox("Select model", list(models.keys()), index=4)
with c_upload:
    uploaded = st.file_uploader("Upload test data (CSV)", type=["csv"])
st.caption(
    "The CSV needs the 12 feature columns and a `good` target column (0/1). "
    "Use the provided `test_data.csv` from the repository."
)

if uploaded is None:
    st.info("⬆️ Upload `test_data.csv` above to see predictions and metrics.")
    with st.expander("Expected feature columns"):
        st.code(", ".join(feature_cols))
    st.stop()

# ---- Read + validate the uploaded data ----
df = pd.read_csv(uploaded)
missing = [c for c in feature_cols if c not in df.columns]
if missing:
    st.error(f"Uploaded CSV is missing required feature columns: {missing}")
    st.stop()
if TARGET not in df.columns:
    st.error(f"Uploaded CSV must contain the target column `{TARGET}` (0/1) for evaluation.")
    st.stop()

st.success(f"Loaded {len(df)} rows. Evaluating **{model_name}**.")
with st.expander("Preview uploaded data"):
    st.dataframe(df.head(10), use_container_width=True)

X = scaler.transform(df[feature_cols].values)
y_true = df[TARGET].values.astype(int)
model = models[model_name]
y_pred = model.predict(X)
y_score = scores_for(model, X)

# ---- (c) Evaluation-metrics display for the SELECTED model ----
st.markdown(f'<div class="step">2 · Evaluation metrics — {model_name}</div>', unsafe_allow_html=True)
metrics = compute_metrics(y_true, y_pred, y_score)
cols = st.columns(6)
for col, (name, val) in zip(cols, metrics.items()):
    col.metric(name, f"{val:.3f}")

# ---- (d) Confusion matrix + classification report for the SELECTED model ----
st.markdown(f'<div class="step">3 · Confusion matrix — {model_name}</div>', unsafe_allow_html=True)
c1, c2 = st.columns([1, 1])
with c1:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.5, 3.6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", cbar=False,
                xticklabels=["not good", "good"], yticklabels=["not good", "good"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
with c2:
    st.text("Classification report")
    st.code(classification_report(y_true, y_pred, target_names=["not good", "good"], zero_division=0))

# ---- All-model comparison, kept OUT of the way in a collapsed expander ----
st.divider()
with st.expander("📊 Compare all 5 models on this test data (click to expand)"):
    st.caption("Metrics for every model on the same uploaded test set — useful for the overall comparison.")
    rows = []
    for name, m in models.items():
        rows.append({"Model": name, **{k: round(v, 4) for k, v in
                                        compute_metrics(y_true, m.predict(X), scores_for(m, X)).items()}})
    st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)

st.caption("UCI Wine Quality (red+white). Models trained on startup — see model/train_models.py.")
