# 🫘 Dry Bean Classifier Lab

Multi-class classification of dry bean varieties from image-derived shape features, with five
classical ML models and an interactive Streamlit app for evaluating them on held-out data.

- **GitHub repository:** https://github.com/belgaonkarsantosh/ml_assignment_2
- **Live Streamlit app:** https://mlassignment2-chahrxyffc4m3shlcgxfqh.streamlit.app/

---

## a. Problem statement

Dry beans are graded by variety before sale, and manual sorting is slow and inconsistent. Given
16 geometric measurements extracted from a photograph of a single bean grain (area, perimeter,
axis lengths, roundness, shape factors, …), predict which of **seven** registered varieties the
grain belongs to: BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER or SIRA.

This is a **multi-class classification** problem. Five classifiers are trained on the same
data and compared using Accuracy, AUC, Precision, Recall, F1 and the Matthews Correlation
Coefficient (MCC).

## b. Dataset description

| Item | Detail |
|---|---|
| Source | UCI Machine Learning Repository — *Dry Bean Dataset* (ID 602) |
| Citation | Koklu, M. & Ozkan, I. A. (2020). Multiclass classification of dry beans using computer vision and machine learning techniques. *Computers and Electronics in Agriculture*, 174, 105507. |
| Instances | 13,611 (13,543 after removing 68 exact duplicate rows) |
| Features | 16 numeric, no missing values |
| Target | `Class` — 7 bean varieties |

**Features** (all computed from the segmented grain image):

| Group | Columns |
|---|---|
| Size | `Area`, `Perimeter`, `ConvexArea`, `EquivDiameter` |
| Axes | `MajorAxisLength`, `MinorAxisLength`, `AspectRation`, `Eccentricity` |
| Shape regularity | `Extent`, `Solidity`, `roundness`, `Compactness` |
| Shape factors | `ShapeFactor1`, `ShapeFactor2`, `ShapeFactor3`, `ShapeFactor4` |

(`AspectRation` is spelled this way in the original dataset and is kept unchanged.)

**Class distribution** (full dataset): DERMASON 3,546 · SIRA 2,636 · SEKER 2,027 · HOROZ 1,928 ·
CALI 1,630 · BARBUNYA 1,322 · BOMBAY 522. The classes are moderately imbalanced, which is why
macro-averaged Precision/Recall/F1 and MCC are reported alongside Accuracy.

**Split:** stratified 80 / 20 train–test split (`random_state=7`) → 10,834 training rows and
2,709 test rows. The test rows are saved as [`test_data.csv`](test_data.csv) and are the data
used for every number in this README and in the app.

## c. GitHub repository link

https://github.com/belgaonkarsantosh/ml_assignment_2

```
dry-bean-classifier-lab/
├── app.py                  # Streamlit app (loads saved models, never retrains)
├── train_models.py         # trains, tunes, evaluates and saves all five models
├── ML_Assignment_2.ipynb   # notebook run on BITS Virtual Lab (calls train_models.py)
├── requirements.txt
├── README.md
├── test_data.csv           # 2,709-row held-out test split (features + Class)
├── data/
│   └── dry_bean.csv        # full UCI dataset converted to CSV
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── svm.joblib
    ├── label_encoder.joblib
    ├── feature_columns.json
    └── metrics.csv
```

## d. Models used

Every model is a scikit-learn `Pipeline` of `StandardScaler` → classifier. Hyperparameters were
tuned with 5-fold stratified cross-validation on the training split only (scoring = macro-F1),
then the best pipeline was evaluated once on the test split.

| Model | Tuned hyperparameters (best) |
|---|---|
| Logistic Regression | `C = 100` (multinomial, `max_iter=3000`) |
| Decision Tree | `max_depth = 8`, `min_samples_leaf = 1` |
| KNN | `n_neighbors = 15`, `weights = "distance"` |
| Naive Bayes (Gaussian) | `var_smoothing = 1e-9` |
| Support Vector Machine | RBF kernel, `C = 10`, `gamma = "scale"` |

### Comparison table (test split, 2,709 rows)

Precision, Recall and F1 are **macro-averaged** over the 7 classes; AUC is **macro one-vs-rest**.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9258 | 0.9946 | 0.9377 | 0.9372 | 0.9374 | 0.9103 |
| Decision Tree | 0.9158 | 0.9791 | 0.9298 | 0.9244 | 0.9270 | 0.8981 |
| KNN | 0.9265 | 0.9925 | 0.9395 | 0.9370 | 0.9381 | 0.9112 |
| Naive Bayes | 0.9018 | 0.9913 | 0.9115 | 0.9112 | 0.9108 | 0.8816 |
| Support Vector Machine | **0.9324** | **0.9953** | **0.9445** | **0.9433** | **0.9439** | **0.9183** |

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong for a linear model (F1 0.937, AUC 0.995). The scaled shape features are close to linearly separable for most varieties, and the weak regularisation chosen by CV (`C = 100`) shows the model benefits from flexible boundaries. Its largest error is DERMASON being predicted as SIRA (44 test rows). |
| Decision Tree | Lowest AUC (0.979) and second-lowest F1 (0.927). A single depth-8 tree makes axis-aligned splits on a few features (`ShapeFactor1`, `ShapeFactor3` and `MajorAxisLength` carry ~77 % of its importance), so its probability estimates are coarse and it struggles most on the SIRA/DERMASON boundary (61 SIRA rows misread as DERMASON — the worst single confusion of any model). Easy to interpret, but the least accurate apart from Naive Bayes. |
| KNN | Nearly ties Logistic Regression (F1 0.938, MCC 0.911). With 15 distance-weighted neighbours on standardised features it captures local, non-linear class structure. Prediction cost grows with the training set because every query scans the stored data. |
| Naive Bayes | Fastest to train (< 1 s) but the weakest overall (Accuracy 0.902, MCC 0.882). Gaussian NB assumes features are independent given the class, which is badly violated here: `Area`, `Perimeter`, `ConvexArea` and `EquivDiameter` are almost perfectly correlated, so the same evidence is counted several times. Its AUC (0.991) is still high, meaning the class *ranking* is good even when the top-1 label is wrong; BARBUNYA suffers most (F1 0.844). |
| Support Vector Machine | Best on every metric (Accuracy 0.932, F1 0.944, MCC 0.918, AUC 0.995). The RBF kernel draws smooth non-linear boundaries, which helps exactly where the other models fail: it has the best F1 for BARBUNYA (0.947), CALI (0.951), SEKER (0.956) and SIRA (0.879). |
| **Overall winner for this dataset** | **Support Vector Machine (RBF).** It leads on all six metrics. The margin over KNN and Logistic Regression is small (~0.6 F1 points), so those are reasonable lighter alternatives. Across all models, BOMBAY is almost perfectly recognised (F1 ≈ 1.00, it is much larger than the others) while **SIRA vs DERMASON** is the hardest pair for every model — the two varieties overlap heavily in size and shape. |

## Streamlit app

The app (`app.py`) provides:

1. **Dataset upload (CSV)** — upload the test split (features + `Class`); a sample
   `test_data.csv` can be downloaded or used directly from the sidebar.
2. **Model selection dropdown** — choose any of the five saved classifiers.
3. **Evaluation metrics** — Accuracy, AUC, Precision, Recall, F1 and MCC for the chosen model.
4. **Confusion matrix and classification report** — counts or row-normalised heatmap, plus
   per-class precision/recall/F1.
5. **Model comparison tab** — all five models scored on the uploaded data, with a metric chart.

The app only loads the saved `.joblib` pipelines and calls `predict` / `predict_proba`; it never
refits on uploaded data.

## How to run locally

```bash
pip install -r requirements.txt
python train_models.py        # optional: retrains and regenerates model/ and test_data.csv
streamlit run app.py
```

`scikit-learn` is pinned to 1.9.0 in `requirements.txt` because the saved models were pickled
with that version; loading them with a different version can fail.
