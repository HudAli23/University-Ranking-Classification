
# URS (University Ranking Classification)

URS is a machine learning project that predicts a university’s **ranking tier** rather than an exact rank. It converts world ranking positions into **10 buckets**, trains supervised models, and outputs reports + plots for analysis.

## Why this project (portfolio framing)

This project demonstrates:

- Practical data cleaning + feature engineering
- Turning a regression-style signal (rank) into a **classification problem**
- Handling **class imbalance** (SMOTE on training only)
- Comparing tree-based models with clear evaluation artifacts

## Dataset

- **File**: `data/shanghai-world-university-ranking (1).xlsx`
- The pipeline performs missing-value imputation and drops unused geo columns.

## Methodology (what happens in the pipeline)

Implemented in `src/URS_pipeline.py`:

1. **Load** data from Excel
2. **Impute missing values**
   - numeric: median
   - categorical: most frequent
3. **Clean rank fields**
   - converts ranges like `"101-150"` to the lower bound integer
4. **Bucketize** world rank into 10 classes:
   - 1: 1–50
   - 2: 51–100
   - …
   - 10: 451+
5. **Train/test split** with stratification
6. **Scale** features (`StandardScaler`)
7. **Balance training set** using SMOTE
8. **Train**:
   - Decision Tree (depth-capped)
   - Random Forest (depth-capped, 50 estimators)
9. **Evaluate**:
   - classification reports
   - confusion matrices
   - metric comparison plot

## Tech Stack

- Python
- Pandas / NumPy
- scikit-learn
- imbalanced-learn (SMOTE)
- Matplotlib / Seaborn

## Setup

```bash
pip install -r requirements.txt
```

## Run

From the `URS` folder:

```bash
python src/URS_pipeline.py
```

## Outputs (generated)

Written to `outputs/`:

- `classification_report_dt_10buckets.txt`
- `classification_report_rf_10buckets.txt`
- `confusion_matrix_dt_10buckets.png`
- `confusion_matrix_rf_10buckets.png`
- `DT_vs_RF_metrics_10buckets.png`

Additional plots may exist under `outputs/confusion_matrices/` and `outputs/metrics_plots/` (repo artifacts).

## Notes / Documentation

- There is a longer write-up in `src/# URS – University Ranking Classification.md` describing the problem framing and evaluation notes.
- Pretrained model artifacts may also be present in `models/`.


## Key contributions (what I built)

- Built an end-to-end, reproducible ML pipeline (load → clean → engineer target → train → evaluate)
- Implemented class-imbalance handling using **SMOTE** (training set only)
- Produced clear evaluation artifacts (reports + confusion matrices + comparison plot)

## Demo in 60 seconds

1. Install deps: `pip install -r requirements.txt`
2. Run: `python src/URS_pipeline.py`
3. Open the generated plots in `outputs/` and explain what they show

## Screenshots

Add screenshots to: `docs/screenshots/`

- `docs/screenshots/confusion_matrix_rf.png` (copy from `outputs/confusion_matrix_rf_10buckets.png`)
- `docs/screenshots/metrics_comparison.png` (copy from `outputs/DT_vs_RF_metrics_10buckets.png`)
