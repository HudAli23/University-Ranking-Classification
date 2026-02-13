import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# 1️⃣ Load dataset
url = r"C:\Users\hudzg\Desktop\protfolio\URS\src\data\shanghai-world-university-ranking (1).xlsx"
df = pd.read_excel(url)

# 2️⃣ Handle missing values
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = df.select_dtypes(include=['object']).columns

numeric_imputer = SimpleImputer(strategy='median')
df[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])

categorical_imputer = SimpleImputer(strategy='most_frequent')
df[categorical_cols] = categorical_imputer.fit_transform(df[categorical_cols])

# Drop irrelevant columns
df.drop(['Geo Shape', 'Geo Point 2D'], axis=1, inplace=True)

# 3️⃣ Encode categorical columns
categorical_columns = ['University', 'Country', 'ISO2 CODE']
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# 4️⃣ Convert World rank to integer
def rank_to_numeric(rank_str):
    try:
        if isinstance(rank_str, str) and '-' in rank_str:
            return int(rank_str.split('-')[0])
        return int(rank_str)
    except:
        return 0

df['World rank integer'] = df['World rank'].apply(rank_to_numeric)

# 5️⃣ Convert National rank to numeric
def convert_range_to_numeric(val):
    try:
        if isinstance(val, str) and '-' in val:
            return float(val.split('-')[0])
        return float(val)
    except:
        return 0

df['National rank'] = df['National rank'].apply(convert_range_to_numeric)

# 6️⃣ Bucket world ranks into 10 classes
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

df['Rank Bucket'] = df['World rank integer'].apply(rank_bucket_10)

# 7️⃣ Prepare features and target
X = df.drop(['World rank', 'World rank integer', 'Rank Bucket'], axis=1)
y = df['Rank Bucket']

# Keep only numeric features for SMOTE
X = X.select_dtypes(include=['float64', 'int64'])

# 8️⃣ Standardize numeric features
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# 9️⃣ Split into training/testing
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

# 1️⃣0️⃣ Apply SMOTE
smote = SMOTE(random_state=42, k_neighbors=2)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# 1️⃣1️⃣ Train models
dt_model = DecisionTreeClassifier(max_depth=10, random_state=42)
rf_model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)

dt_model.fit(X_train_res, y_train_res)
rf_model.fit(X_train_res, y_train_res)

# 1️⃣2️⃣ Training vs test accuracy
print("Decision Tree training accuracy:", dt_model.score(X_train_res, y_train_res))
print("Decision Tree test accuracy:", dt_model.score(X_test, y_test))
print("Random Forest training accuracy:", rf_model.score(X_train_res, y_train_res))
print("Random Forest test accuracy:", rf_model.score(X_test, y_test))

# 1️⃣3️⃣ Make predictions
dt_preds = dt_model.predict(X_test)
rf_preds = rf_model.predict(X_test)

# 1️⃣4️⃣ Classification reports
dt_report = classification_report(y_test, dt_preds)
rf_report = classification_report(y_test, rf_preds)

print("Decision Tree Classification Report:\n", dt_report)
print("Random Forest Classification Report:\n", rf_report)

# Save reports
with open("classification_report_dt_10buckets.txt", "w") as f:
    f.write(dt_report)
with open("classification_report_rf_10buckets.txt", "w") as f:
    f.write(rf_report)

# 1️⃣5️⃣ Confusion matrices
plt.figure(figsize=(10,7))
sns.heatmap(confusion_matrix(y_test, dt_preds), annot=True, fmt='d', cmap='Blues')
plt.title('Decision Tree Confusion Matrix (10 Buckets)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig("confusion_matrix_dt_10buckets.png")
plt.show()

plt.figure(figsize=(10,7))
sns.heatmap(confusion_matrix(y_test, rf_preds), annot=True, fmt='d', cmap='Blues')
plt.title('Random Forest Confusion Matrix (10 Buckets)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig("confusion_matrix_rf_10buckets.png")
plt.show()

# =========================
# 1️⃣6️⃣ Bar chart comparison (PUT HERE)
# =========================

import numpy as np

# Metrics to compare
metrics = ['precision', 'recall', 'f1-score']

# Convert classification reports to dictionaries
dt_metrics = classification_report(y_test, dt_preds, output_dict=True)
rf_metrics = classification_report(y_test, rf_preds, output_dict=True)

# Prepare data for plotting
dt_values = [dt_metrics['macro avg'][metric] for metric in metrics] + [dt_model.score(X_test, y_test)]
rf_values = [rf_metrics['macro avg'][metric] for metric in metrics] + [rf_model.score(X_test, y_test)]

# Create DataFrame for plotting
metrics_df = pd.DataFrame({
    'Decision Tree': dt_values,
    'Random Forest': rf_values
}, index=metrics + ['accuracy'])

# Plot bar chart
plt.figure(figsize=(10,6))
metrics_df.plot(kind='bar', figsize=(10,6), color=['#1f77b4', '#ff7f0e'])
plt.title('Decision Tree vs Random Forest: 10-Bucket Model Metrics')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("DT_vs_RF_metrics_10buckets.png")
plt.show()
