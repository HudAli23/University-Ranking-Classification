"""Pipeline: load ranking data -> clean/impute -> bucketize ranks -> scale -> SMOTE -> train DT/RF -> evaluate and save reports/plots."""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from imblearn.over_sampling import SMOTE
from pathlib import Path


# =========================
# PATH CONFIGURATION
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "shanghai-world-university-ranking (1).xlsx"

OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.3


# =========================
# LOAD DATA
# =========================
df = pd.read_excel(DATA_PATH)


# =========================
# HANDLE MISSING VALUES
# =========================
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
categorical_cols = df.select_dtypes(include=["object"]).columns

df[numeric_cols] = SimpleImputer(strategy="median").fit_transform(df[numeric_cols])
df[categorical_cols] = SimpleImputer(strategy="most_frequent").fit_transform(df[categorical_cols])

df.drop(["Geo Shape", "Geo Point 2D"], axis=1, inplace=True)


# =========================
# CONVERT RANK FIELDS
# =========================
def rank_to_numeric(val):
    if isinstance(val, str) and "-" in val:
        return int(val.split("-")[0])
    try:
        return int(val)
    except:
        return np.nan

df["World rank integer"] = df["World rank"].apply(rank_to_numeric)


def national_rank_to_numeric(val):
    if isinstance(val, str) and "-" in val:
        return float(val.split("-")[0])
    try:
        return float(val)
    except:
        return np.nan

df["National rank"] = df["National rank"].apply(national_rank_to_numeric)


# =========================
# CREATE 10 RANK BUCKETS
# =========================
def rank_bucket_10(rank):
    if rank <= 50: return 1
    elif rank <= 100: return 2
    elif rank <= 150: return 3
    elif rank <= 200: return 4
    elif rank <= 250: return 5
    elif rank <= 300: return 6
    elif rank <= 350: return 7
    elif rank <= 400: return 8
    elif rank <= 450: return 9
    else: return 10

df["Rank Bucket"] = df["World rank integer"].apply(rank_bucket_10)


# =========================
# FEATURES & TARGET
# =========================
X = df.drop(["World rank", "World rank integer", "Rank Bucket"], axis=1)
y = df["Rank Bucket"]

X = X.select_dtypes(include=["int64", "float64"])


# =========================
# TRAIN / TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)


# =========================
# SCALE FEATURES
# =========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# =========================
# APPLY SMOTE
# =========================
smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=2)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)


# =========================
# TRAIN MODELS
# =========================
dt_model = DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE)
rf_model = RandomForestClassifier(
    n_estimators=50,
    max_depth=10,
    random_state=RANDOM_STATE
)

dt_model.fit(X_train_res, y_train_res)
rf_model.fit(X_train_res, y_train_res)


# =========================
# EVALUATION
# =========================
dt_preds = dt_model.predict(X_test_scaled)
rf_preds = rf_model.predict(X_test_scaled)

dt_report = classification_report(y_test, dt_preds)
rf_report = classification_report(y_test, rf_preds)

with open(OUTPUT_DIR / "classification_report_dt_10buckets.txt", "w") as f:
    f.write(dt_report)

with open(OUTPUT_DIR / "classification_report_rf_10buckets.txt", "w") as f:
    f.write(rf_report)


# =========================
# CONFUSION MATRICES
# =========================
plt.figure(figsize=(10, 7))
sns.heatmap(confusion_matrix(y_test, dt_preds), annot=True, fmt="d", cmap="Blues")
plt.title("Decision Tree Confusion Matrix (10 Buckets)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix_dt_10buckets.png")
plt.show()

plt.figure(figsize=(10, 7))
sns.heatmap(confusion_matrix(y_test, rf_preds), annot=True, fmt="d", cmap="Blues")
plt.title("Random Forest Confusion Matrix (10 Buckets)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix_rf_10buckets.png")
plt.show()


# =========================
# METRIC COMPARISON PLOT
# =========================
metrics = ["precision", "recall", "f1-score"]

dt_metrics = classification_report(y_test, dt_preds, output_dict=True)
rf_metrics = classification_report(y_test, rf_preds, output_dict=True)

metrics_df = pd.DataFrame({
    "Decision Tree": [dt_metrics["macro avg"][m] for m in metrics] + [dt_model.score(X_test_scaled, y_test)],
    "Random Forest": [rf_metrics["macro avg"][m] for m in metrics] + [rf_model.score(X_test_scaled, y_test)]
}, index=metrics + ["accuracy"])

metrics_df.plot(kind="bar", figsize=(10, 6))
plt.title("Decision Tree vs Random Forest (10-Bucket Classification)")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "DT_vs_RF_metrics_10buckets.png")
plt.show()


print(f"✅ All outputs saved to: {OUTPUT_DIR.resolve()}")
