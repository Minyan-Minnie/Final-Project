import warnings
warnings.filterwarnings("ignore")  # suppress warnings for cleaner output

import pandas as pd
import numpy as np
import sqlite3

from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import Reweighing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from fairlearn.metrics import demographic_parity_difference
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# 1. Load raw data
df = pd.read_csv("adult.csv")

# 2. Replace '?' with NaN and drop missing rows
df.replace('?', np.nan, inplace=True)
df.dropna(inplace=True)

# 3. Encode 'sex' as binary numeric BEFORE one-hot encoding
df['sex'] = df['sex'].apply(lambda x: 1 if x.strip() == 'Male' else 0)

# 4. List categorical columns to one-hot encode (except 'sex' which is numeric now)
categorical_cols = [
    'workclass', 'education', 'marital.status', 'occupation',
    'relationship', 'race', 'native.country'
]

# 5. One-hot encode categorical variables
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# 6. Encode 'income' target as binary label
df_encoded['income'] = df_encoded['income'].apply(lambda x: 1 if '>50K' in x else 0)

# 7. Prepare BinaryLabelDataset for AIF360 fairness tools
label_name = 'income'
protected_attribute = 'sex'

dataset = BinaryLabelDataset(df=df_encoded,
                             label_names=[label_name],
                             protected_attribute_names=[protected_attribute])

# 8. Split dataset into train/test (fixed random seed for reproducibility)
dataset_train, dataset_test = dataset.split([0.8], shuffle=True, seed=42)

# 9. Reweighing fairness preprocessing
rw = Reweighing(unprivileged_groups=[{protected_attribute: 0}],
                privileged_groups=[{protected_attribute: 1}])
rw.fit(dataset_train)
dataset_train_rw = rw.transform(dataset_train)

# 10. Train logistic regression on reweighted training data
clf_rw = LogisticRegression(max_iter=1000, solver='lbfgs')
clf_rw.fit(dataset_train_rw.features, dataset_train_rw.labels.ravel(),
           sample_weight=dataset_train_rw.instance_weights)

# 11. Evaluate reweighted model on test set
y_pred_rw = clf_rw.predict(dataset_test.features)
print(f"F1 Score (Reweighing): {f1_score(dataset_test.labels.ravel(), y_pred_rw):.3f}")
print(f"Demographic Parity Difference (gender): {demographic_parity_difference(dataset_test.labels.ravel(), y_pred_rw, sensitive_features=dataset_test.protected_attributes.ravel()):.3f}")

# 12. SMOTE oversampling (no scaling yet)
X_train = dataset_train.features
y_train = dataset_train.labels.ravel()
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

clf_smote = LogisticRegression(max_iter=1000, solver='lbfgs')
clf_smote.fit(X_train_sm, y_train_sm)

y_pred_smote = clf_smote.predict(dataset_test.features)
print(f"F1 Score with SMOTE: {f1_score(dataset_test.labels.ravel(), y_pred_smote):.3f}")
print(f"Demographic Parity Difference with SMOTE: {demographic_parity_difference(dataset_test.labels.ravel(), y_pred_smote, sensitive_features=dataset_test.protected_attributes.ravel()):.3f}")

# 13. Feature scaling (standardization)
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(dataset_test.features)

# Train on scaled + reweighted data
clf_rw_scaled = LogisticRegression(max_iter=2000, solver='lbfgs')
clf_rw_scaled.fit(X_train_scaled, y_train, sample_weight=dataset_train_rw.instance_weights)

y_pred_rw_scaled = clf_rw_scaled.predict(X_test_scaled)
print(f"F1 Score (Reweighing + Scaling): {f1_score(dataset_test.labels.ravel(), y_pred_rw_scaled):.3f}")
print(f"Demographic Parity Difference (gender): {demographic_parity_difference(dataset_test.labels.ravel(), y_pred_rw_scaled, sensitive_features=dataset_test.protected_attributes.ravel()):.3f}")

# 14. SMOTE + Scaling
X_train_sm_scaled = scaler.transform(X_train_sm)
clf_smote_scaled = LogisticRegression(max_iter=2000, solver='lbfgs')
clf_smote_scaled.fit(X_train_sm_scaled, y_train_sm)

y_pred_smote_scaled = clf_smote_scaled.predict(X_test_scaled)
print(f"F1 Score (SMOTE + Scaling): {f1_score(dataset_test.labels.ravel(), y_pred_smote_scaled):.3f}")
print(f"Demographic Parity Difference (SMOTE + Scaling): {demographic_parity_difference(dataset_test.labels.ravel(), y_pred_smote_scaled, sensitive_features=dataset_test.protected_attributes.ravel()):.3f}")

# 15. Baseline logistic regression (no mitigation)
clf_baseline = LogisticRegression(max_iter=1000, solver='lbfgs')
clf_baseline.fit(X_train, y_train)
y_pred_baseline = clf_baseline.predict(dataset_test.features)
print(f"Baseline Logistic Regression F1 Score: {f1_score(dataset_test.labels.ravel(), y_pred_baseline):.3f}")
print(f"Baseline Demographic Parity Difference (gender): {demographic_parity_difference(dataset_test.labels.ravel(), y_pred_baseline, sensitive_features=dataset_test.protected_attributes.ravel()):.3f}")

# 16. Save datasets to SQLite for later analysis
conn = sqlite3.connect('adult_fairness.db')

# Original train data
train_df = pd.DataFrame(dataset_train.features, columns=dataset_train.feature_names)
train_df[label_name] = dataset_train.labels.ravel()
train_df[protected_attribute] = dataset_train.protected_attributes.ravel()
train_df.to_sql('train_original', conn, if_exists='replace', index=False)

# Reweighted train data (including instance weights)
train_rw_df = pd.DataFrame(dataset_train_rw.features, columns=dataset_train_rw.feature_names)
train_rw_df[label_name] = dataset_train_rw.labels.ravel()
train_rw_df[protected_attribute] = dataset_train_rw.protected_attributes.ravel()
train_rw_df['instance_weights'] = dataset_train_rw.instance_weights
train_rw_df.to_sql('train_reweighted', conn, if_exists='replace', index=False)

# Test set
test_df = pd.DataFrame(dataset_test.features, columns=dataset_test.feature_names)
test_df[label_name] = dataset_test.labels.ravel()
test_df[protected_attribute] = dataset_test.protected_attributes.ravel()
test_df.to_sql('test_set', conn, if_exists='replace', index=False)

conn.close()
