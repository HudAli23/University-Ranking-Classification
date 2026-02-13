# URS – University Ranking Classification System

## Overview
URS is a machine learning project that classifies universities into ranking tiers using
data from the Shanghai World University Rankings. The project transforms continuous
ranking positions into discrete ranking buckets and evaluates supervised learning models
on their ability to predict these tiers.

---

## Problem Statement
University rankings are influenced by multiple academic and institutional indicators.
Rather than predicting an exact rank, this project reframes the task as a multi-class
classification problem by grouping universities into ranking tiers.

The objectives are to:
- Convert raw ranking data into structured classes
- Address class imbalance in ranking distributions
- Compare tree-based classification models

---

## Dataset
- **Source:** Shanghai World University Rankings
- **Includes:**
  - World and national ranking positions
  - Academic performance indicators
  - Institutional metadata
- Missing values are handled using statistical imputation.

---

## Methodology

### 1. Data Cleaning
- Removed unused geographical fields
- Converted ranking ranges (e.g. `"101–150"`) into numeric values
- Standardized numeric inputs

### 2. Target Engineering
World rankings were grouped into **10 ranking buckets** using fixed cutoffs:

| Bucket | World Rank Range |
|------|------------------|
| 1 | 1–50 |
| 2 | 51–100 |
| 3 | 101–150 |
| 4 | 151–200 |
| 5 | 201–250 |
| 6 | 251–300 |
| 7 | 301–350 |
| 8 | 351–400 |
| 9 | 401–450 |
| 10 | 451+ |

This converts the task into a multi-class classification problem.

### 3. Feature Processing
- Median imputation for numeric features
- Feature scaling using `StandardScaler`
- Only numeric features were retained for modeling

### 4. Class Imbalance Handling
- SMOTE applied to the **training set only** to balance ranking classes

### 5. Models Trained
- Decision Tree Classifier
- Random Forest Classifier

All experiments use a fixed `random_state` to ensure reproducibility.

---

## Evaluation
Models were evaluated using:
- Precision, Recall, and F1-score (macro-averaged)
- Accuracy
- Confusion matrices for class-level performance analysis

### Key Findings
- Random Forest achieved higher macro F1-score and accuracy
- Decision Tree showed signs of overfitting
- Middle-ranking buckets were the most challenging to classify

---

## Technologies Used
- Python
- pandas, numpy
- scikit-learn
- imbalanced-learn
- matplotlib, seaborn

---

## Project Structure
URS/
│
├── data/
│ └── shanghai_world_university_rankings.xlsx
├── src/
│ └── URS_pipeline.py
├── outputs/
│ ├── classification_report_dt_10buckets.txt
│ ├── classification_report_rf_10buckets.txt
│ ├── confusion_matrix_dt_10buckets.png
│ ├── confusion_matrix_rf_10buckets.png
│ └── DT_vs_RF_metrics_10buckets.png
└── README.md