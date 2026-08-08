# Wine Quality Classification — Model Comparison & Streamlit App

An end-to-end machine-learning project that trains and compares **five
classification models** on the UCI Wine Quality dataset and serves them through
an interactive **Streamlit** web app.

---

## a. Problem statement

Given the physicochemical properties of a wine (acidity, sugar, sulphates,
alcohol, etc.), predict whether the wine is **good** — defined as a sensory
quality rating of **7 or higher** on the original 0–10 scale. This is framed as a
**binary classification** problem (`good = 1`, `not good = 0`).

Being able to flag high-quality wines automatically from cheap chemical
measurements is useful for quality control in production, where expert sensory
panels are slow and expensive.

## b. Dataset description

- **Source:** UCI Machine Learning Repository — *Wine Quality Data Set*
  (`winequality-red.csv` + `winequality-white.csv`).
- **Construction:** the red and white CSVs are concatenated, and a binary
  `wine_type` feature is added (`0 = red`, `1 = white`).
- **Instances:** **6,497** (1,599 red + 4,898 white) — well above the 500 minimum.
- **Features:** **12** — 11 physicochemical measurements plus `wine_type`:
  `fixed acidity`, `volatile acidity`, `citric acid`, `residual sugar`,
  `chlorides`, `free sulfur dioxide`, `total sulfur dioxide`, `density`, `pH`,
  `sulphates`, `alcohol`, `wine_type`.
- **Target:** `good` = 1 if original quality ≥ 7, else 0.
- **Class balance:** the positive ("good") class is the minority at **≈19.7%**,
  which makes threshold-independent metrics (AUC) and the balanced correlation
  metric (MCC) more informative than raw accuracy.
- **Preprocessing:** stratified **80/20** train/test split, followed by
  `StandardScaler` standardisation fitted on the training set only (no leakage).

## c. GitHub repository link

`https://github.com/<your-username>/wine-quality-classifier`  <!-- replace with your repo URL -->

Repository contents:

```
wine-quality-classifier/
├── app.py                   # Streamlit app (upload CSV, pick model, view metrics + confusion matrix)
├── requirements.txt         # deployment dependencies
├── README.md                # this file
├── test_data.csv            # held-out test set (1,300 rows) for the app
├── .gitignore
└── model/
    ├── model_training.ipynb # notebook: EDA, trains 5 models, 6 metrics, confusion matrices + ROC curves
    ├── train_models.py      # script version of the training (same logic, headless)
    ├── scaler.joblib
    ├── feature_cols.joblib
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    └── random_forest_ensemble.joblib
```

## d. Models used

Five classifiers were trained on the **same** dataset and evaluated on the
identical held-out test set (1,300 rows). All metrics below are computed on the
test set.

| ML Model Name           | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
|-------------------------|----------|--------|-----------|--------|--------|--------|
| Logistic Regression     | 0.8223   | 0.8048 | 0.6147    | 0.2617 | 0.3671 | 0.3178 |
| Decision Tree           | 0.8354   | 0.8180 | 0.6000    | 0.4922 | 0.5408 | 0.4449 |
| kNN                     | 0.8246   | 0.8306 | 0.5745    | 0.4219 | 0.4865 | 0.3904 |
| Naive Bayes             | 0.7346   | 0.7486 | 0.3901    | 0.6172 | 0.4781 | 0.3268 |
| Random Forest (Ensemble)| **0.8892** | **0.9165** | **0.8077** | **0.5742** | **0.6712** | **0.6197** |

*(Metrics are reproducible by running `python model/train_models.py`, `random_state=42`.)*

### Observations on model performance

| ML Model Name            | Observation about model performance |
|--------------------------|-------------------------------------|
| **Logistic Regression**  | Reasonable accuracy (0.82) but the **lowest recall (0.26)** — its single linear boundary cannot capture the non-linear chemistry of "good" wines, so it misses most positives. High precision, low recall: it only flags wines it is very sure about. |
| **Decision Tree**        | Best of the simple models on F1/MCC (0.54 / 0.44). Non-linear splits recover far more positives than logistic regression (recall 0.49). Depth was capped at 8 to limit overfitting. |
| **kNN**                  | Middle-of-the-pack. Good AUC (0.83) because distance ranking separates classes well, but recall (0.42) suffers as the minority "good" points are outnumbered by neighbours from the majority class. |
| **Naive Bayes**          | **Highest recall (0.62)** but **lowest precision (0.39)** and lowest accuracy. Its feature-independence assumption is violated (e.g. the two sulfur-dioxide features and density/alcohol are correlated), so it over-predicts the positive class. |
| **Random Forest (Ensemble)** | **Clear winner on every metric** — accuracy 0.89, AUC 0.92, F1 0.67, MCC 0.62. Averaging many de-correlated trees captures non-linear interactions while controlling variance, giving both the best precision and a strong recall. |
| **Overall winner for your dataset?** | **Random Forest (Ensemble).** It dominates all six metrics, and its lead is largest on the imbalance-sensitive AUC and MCC — the most trustworthy scores given the ≈20% positive rate. |

**Key takeaway:** on this dataset accuracy alone is misleading (a "predict-not-good"
baseline already scores ≈0.80). AUC and MCC reveal the real gap between a linear
model and the ensemble, and confirm Random Forest as the best choice.

## Streamlit app

The deployed app lets you:

- **Upload** a test CSV (use the provided `test_data.csv`).
- **Select** any of the five models from a dropdown.
- **View** all six evaluation metrics for the chosen model, plus a table of every
  model on the uploaded data.
- **Inspect** the confusion matrix and full classification report.

Live app: `https://<your-app>.streamlit.app`  <!-- replace with your deployed URL -->

### Run locally

```bash
pip install -r requirements.txt
python model/train_models.py     # trains models + writes test_data.csv (one-time)
streamlit run app.py
```

## Tech stack

Python, scikit-learn, pandas, NumPy, matplotlib, seaborn, Streamlit, joblib.
