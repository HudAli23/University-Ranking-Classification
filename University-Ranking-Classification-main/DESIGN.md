# URS (University Ranking Classification) — Software Design Document

## 1. Overview

**URS** is a machine learning project that predicts a university's **ranking tier** (bucket) rather than an exact rank. It converts world ranking positions into **10 buckets**, trains supervised models (Decision Tree and Random Forest), and outputs classification reports, confusion matrices, and metric comparison plots.

| Attribute | Value |
|-----------|--------|
| **Project type** | ML pipeline (batch script) |
| **Language** | Python |
| **Libraries** | Pandas, NumPy, scikit-learn, imbalanced-learn (SMOTE), Matplotlib, Seaborn |
| **Entry point** | `src/URS_pipeline.py` |
| **Output** | Reports and plots in `outputs/`; models in `models/` (if saved) |

---

## 2. Goals and Scope

- **Primary goal**: Turn a regression-style signal (world rank) into a **classification problem** (10 buckets) and compare tree-based models with clear evaluation artifacts.
- **Scope**: Load Excel data, impute missing values, clean rank fields, bucketize world rank, stratified train/test split, scale features, balance training set (SMOTE), train DT and RF, evaluate (reports, confusion matrices, metric comparison).
- **Out of scope**: Real-time API, web UI, deployment as a service, hyperparameter tuning automation.

---

## 3. System Context

- **Users**: Data scientist / analyst; run pipeline from command line.
- **Input**: Excel file `data/shanghai-world-university-ranking (1).xlsx`.
- **Output**: Text reports and PNG plots in `outputs/`; optionally saved model artifacts in `models/`.
- **Deployment**: Local execution; `pip install -r requirements.txt` then `python src/URS_pipeline.py`.

---

## 4. Architecture

### 4.1 Pipeline Stages (in order)

1. **Load** — Read Excel into a Pandas DataFrame.
2. **Impute** — Numeric: median; categorical: most frequent. Drop unused geo columns (e.g. Geo Shape, Geo Point 2D).
3. **Clean rank fields** — Convert ranges like `"101-150"` to lower-bound integer; derive `World rank integer` and cleaned `National rank`.
4. **Bucketize** — Map world rank to 10 classes:
   - 1: 1–50, 2: 51–100, 3: 101–150, 4: 151–200, 5: 201–250, 6: 251–300, 7: 301–350, 8: 351–400, 9: 401–450, 10: 451+.
5. **Features and target** — Target: `Rank Bucket`. Features: all remaining numeric columns (excluding rank raw/bucket).
6. **Train/test split** — Stratified by target; test size 0.3; fixed random state.
7. **Scale** — `StandardScaler` fit on train, transform train and test.
8. **Balance** — SMOTE on training set only (e.g. k_neighbors=2); test set unchanged.
9. **Train** — Decision Tree (max_depth=10), Random Forest (n_estimators=50, max_depth=10).
10. **Evaluate** — Classification reports (macro avg etc.), confusion matrices (heatmaps), DT vs RF metric comparison bar chart.
11. **Save** — Reports to `.txt`; plots to `.png` in `outputs/`; optionally save models (e.g. `.pkl`).

### 4.2 Data Flow (Conceptual)

- **Input**: Single Excel file (Shanghai world university ranking).
- **Intermediate**: DataFrame after impute/clean/bucket; X_train, X_test, y_train, y_test; scaled and SMOTE-resampled training data; fitted DT and RF.
- **Output**: `classification_report_dt_10buckets.txt`, `classification_report_rf_10buckets.txt`, `confusion_matrix_dt_10buckets.png`, `confusion_matrix_rf_10buckets.png`, `DT_vs_RF_metrics_10buckets.png` (and any extra under `outputs/` or `models/`).

### 4.3 Key Design Choices

| Decision | Rationale |
|----------|------------|
| 10 buckets | Coarse enough for interpretability; fine enough to show model differences. |
| SMOTE on train only | Avoid data leakage; test set reflects real class distribution. |
| Stratified split | Preserve class proportions in train/test. |
| StandardScaler | Tree models are scale-invariant but scaling can help if pipeline is extended (e.g. other algorithms). |
| Depth-capped trees | Reduce overfitting; comparable complexity between DT and RF. |

---

## 5. Tech Stack (Summary)

- **Python** 3.x
- **Pandas** — Load and clean Excel.
- **NumPy** — Numeric operations.
- **scikit-learn** — Imputer, StandardScaler, train_test_split, DecisionTreeClassifier, RandomForestClassifier, classification_report, confusion_matrix.
- **imbalanced-learn** — SMOTE.
- **Matplotlib / Seaborn** — Confusion matrix heatmaps and metric comparison plot.

---

## 6. Directory Layout

- **data/** — Input Excel file.
- **src/** — `URS_pipeline.py`; optional markdown doc.
- **outputs/** — Generated reports and plots.
- **models/** — Optional saved model artifacts (e.g. `.pkl`).
- **requirements.txt** — Python dependencies.

---

## 7. Diagram Reference

- **How the project works**: See `URS-Pipeline.puml` (PlantUML). The diagram is an **activity diagram** showing the ML pipeline step by step: run script → load Excel → impute → clean ranks → bucketize → split → scale → SMOTE → train DT and RF → evaluate → save reports and plots to `outputs/`. Use a PlantUML renderer to generate the image.
