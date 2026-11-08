"""Streamlit front end for the Dry Bean classifiers (evaluation only, no retraining)."""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
TARGET = "Class"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}
METRIC_NAMES = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]

st.set_page_config(page_title="Dry Bean Classifier Lab", page_icon="🫘", layout="wide")

st.markdown(
    """
    <style>
    .bean-hero {padding: 1.1rem 1.4rem; border-radius: 14px;
                background: linear-gradient(120deg, #5b3a1e 0%, #a86b32 55%, #e3b778 100%);
                color: #fffaf2; margin-bottom: 1rem;}
    .bean-hero h1 {margin: 0; font-size: 2rem;}
    .bean-hero p {margin: .3rem 0 0 0; opacity: .92;}
    div[data-testid="stMetric"] {background: #fbf4ea; border: 1px solid #ead7bd;
                                 border-radius: 12px; padding: .6rem .8rem;}
    </style>
    <div class="bean-hero">
      <h1>🫘 Dry Bean Classifier Lab</h1>
      <p>Upload held-out bean measurements, pick a trained model, and inspect how well it
      separates the seven bean varieties. Models are pre-trained; nothing is refit here.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifacts():
    models = {name: joblib.load(MODEL_DIR / fname) for name, fname in MODEL_FILES.items()}
    encoder = joblib.load(MODEL_DIR / "label_encoder.joblib")
    features = json.loads((MODEL_DIR / "feature_columns.json").read_text())
    train_metrics = pd.read_csv(MODEL_DIR / "metrics.csv")
    return models, encoder, features, train_metrics


def macro_ovr_auc(y_true, proba, n_classes):
    """Macro one-vs-rest AUC over classes that have both positives and negatives in y_true."""
    aucs = []
    for k in range(n_classes):
        positives = y_true == k
        if 0 < positives.sum() < len(y_true):
            aucs.append(roc_auc_score(positives, proba[:, k]))
    return float(np.mean(aucs)) if aucs else float("nan")


def score_model(model, X, y_true, n_classes):
    y_pred = model.predict(X)
    proba = model.predict_proba(X)
    scores = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": macro_ovr_auc(y_true, proba, n_classes),
        "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
    return scores, y_pred


models, encoder, FEATURES, train_metrics = load_artifacts()
CLASSES = list(encoder.classes_)

with st.sidebar:
    st.header("1 · Test data")
    uploaded = st.file_uploader("Upload test CSV (features + Class)", type=["csv"])
    sample_path = ROOT / "test_data.csv"
    if sample_path.exists():
        st.download_button(
            "Download sample test_data.csv",
            data=sample_path.read_bytes(),
            file_name="test_data.csv",
            mime="text/csv",
        )
    use_sample = st.checkbox("Use bundled test_data.csv instead", value=uploaded is None)

    st.header("2 · Model")
    model_name = st.selectbox("Choose a classifier", list(MODEL_FILES))
    st.caption(
        "Precision, Recall and F1 are macro-averaged across the 7 classes; "
        "AUC is macro one-vs-rest."
    )

if uploaded is not None and not use_sample:
    data = pd.read_csv(uploaded)
    source = f"uploaded file `{uploaded.name}`"
elif use_sample and sample_path.exists():
    data = pd.read_csv(sample_path)
    source = "bundled `test_data.csv`"
else:
    st.info("Upload a test CSV in the sidebar (or tick the bundled-data option) to begin.")
    st.stop()

missing = [c for c in FEATURES + [TARGET] if c not in data.columns]
if missing:
    st.error(f"The CSV is missing required columns: {', '.join(missing)}")
    st.stop()

data = data.dropna(subset=FEATURES + [TARGET])
unknown = sorted(set(data[TARGET]) - set(CLASSES))
if unknown:
    st.error(f"Unknown class labels in `{TARGET}`: {unknown}. Expected one of {CLASSES}.")
    st.stop()

X = data[FEATURES]
y_true = encoder.transform(data[TARGET])
st.caption(f"Evaluating on {source} — {len(data):,} rows, {data[TARGET].nunique()} classes present.")

tab_eval, tab_compare, tab_data = st.tabs(
    ["📊 Selected model", "🏁 Compare all models", "🔎 Data preview"]
)

with tab_eval:
    scores, y_pred = score_model(models[model_name], X, y_true, len(CLASSES))
    st.subheader(f"{model_name} — evaluation metrics")
    cols = st.columns(len(METRIC_NAMES))
    for col, metric in zip(cols, METRIC_NAMES):
        value = scores[metric]
        col.metric(metric, "n/a" if np.isnan(value) else f"{value:.4f}")

    left, right = st.columns([1.1, 1])
    with left:
        st.markdown("**Confusion matrix**")
        normalize = st.toggle("Show row-normalised (recall per class)", value=False)
        cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASSES)))
        if normalize:
            with np.errstate(invalid="ignore", divide="ignore"):
                cm_plot = np.nan_to_num(cm / cm.sum(axis=1, keepdims=True))
            fmt = ".2f"
        else:
            cm_plot, fmt = cm, "d"
        fig, ax = plt.subplots(figsize=(6.4, 5.2))
        sns.heatmap(
            cm_plot, annot=True, fmt=fmt, cmap="YlOrBr", cbar=False,
            xticklabels=CLASSES, yticklabels=CLASSES, ax=ax,
        )
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.markdown("**Classification report**")
        report = classification_report(
            y_true, y_pred, labels=range(len(CLASSES)), target_names=CLASSES,
            output_dict=True, zero_division=0,
        )
        report_df = pd.DataFrame(report).T
        report_df["support"] = report_df["support"].astype(int)
        st.dataframe(
            report_df.style.format({"precision": "{:.3f}", "recall": "{:.3f}",
                                    "f1-score": "{:.3f}", "support": "{:,}"})
            .background_gradient(subset=["f1-score"], cmap="YlOrBr"),
            width="stretch",
        )
        worst = report_df.loc[CLASSES, "f1-score"].idxmin()
        st.caption(f"Hardest class for this model on this data: **{worst}**.")

with tab_compare:
    st.subheader("All five models on the same test data")
    rows = []
    for name, model in models.items():
        s, _ = score_model(model, X, y_true, len(CLASSES))
        rows.append({"ML Model Name": name, **s})
    compare = pd.DataFrame(rows).set_index("ML Model Name")
    best = compare["F1"].idxmax()
    st.dataframe(
        compare.style.format("{:.4f}").highlight_max(axis=0, color="#f2d3a3"),
        width="stretch",
    )
    st.success(f"Best macro-F1 on this data: **{best}** ({compare.loc[best, 'F1']:.4f}).")

    metric_pick = st.radio("Metric to chart", METRIC_NAMES, horizontal=True, index=4)
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ordered = compare[metric_pick].sort_values()
    ax.barh(ordered.index, ordered.values, color="#a86b32")
    lo = np.nanmin(ordered.values)
    ax.set_xlim(max(0.0, lo - 0.05), 1.0)
    for i, v in enumerate(ordered.values):
        ax.text(v, i, f" {v:.4f}", va="center", fontsize=9)
    ax.set_xlabel(metric_pick)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    with st.expander("Hold-out metrics recorded at training time"):
        st.dataframe(train_metrics.set_index("ML Model Name"), width="stretch")

with tab_data:
    st.subheader("Uploaded data")
    st.dataframe(data.head(50), width="stretch")
    counts = data[TARGET].value_counts().reindex(CLASSES, fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(counts.index, counts.values, color="#5b3a1e")
    ax.set_ylabel("Rows")
    ax.set_title("Class distribution in uploaded data")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
