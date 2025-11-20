# --- Cell 1: Import Libraries ---
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import random
import time
from collections import Counter

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    f1_score,
    precision_score,
    recall_score,
    brier_score_loss,
    roc_auc_score,
    precision_recall_curve,
    auc
)

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

print("Libraries imported successfully.")

# --- Cell 2: Data Loading and Splitting ---
# Load data and split into training and test sets BEFORE scaling
# to avoid Data Leakage.

df = pd.read_csv('creditcard.csv')

X = df.drop('Class', axis=1)
y = df['Class']

# 80% training, 20% test split with stratification (maintaining class proportions)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

print("Data split complete.")
print(f"Class distribution in training set: {Counter(y_train)}")
print("-" * 50)

# --- Cell 3: Preprocessing (Scaling) ---
# Use RobustScaler, which is resistant to outliers.
# Fit the scaler ONLY on training data.

# Scale 'Amount' column
scaler_amount = RobustScaler().fit(X_train[['Amount']])
X_train['scaled_amount'] = scaler_amount.transform(X_train[['Amount']])
X_test['scaled_amount'] = scaler_amount.transform(X_test[['Amount']])

# Scale 'Time' column
scaler_time = RobustScaler().fit(X_train[['Time']])
X_train['scaled_time'] = scaler_time.transform(X_train[['Time']])
X_test['scaled_time'] = scaler_time.transform(X_test[['Time']])

# Drop original, unscaled columns
X_train = X_train.drop(['Time', 'Amount'], axis=1)
X_test = X_test.drop(['Time', 'Amount'], axis=1)

print("Data scaling complete.")
print("-" * 50)

# --- Cell 4: Model Configuration and Hyperparameters ---
# Calculate minority class weight (scale_pos_weight) to balance class impact.

neg_count = Counter(y_train)[0]
pos_count = Counter(y_train)[1]
scale_pos_weight_value = neg_count / pos_count

print(f"Calculated scale_pos_weight: {scale_pos_weight_value:.2f}")

# Hyperparameters found during tuning process (Milestone 5)
best_params_xgb = {
    'subsample': 0.6,
    'reg_lambda': 1,
    'reg_alpha': 0.005,
    'n_estimators': 300,
    'min_child_weight': 1,
    'max_depth': 7,
    'learning_rate': 0.1,
    'gamma': 0.0,
    'colsample_bytree': 0.6
}

print("Optimal hyperparameters loaded.")
print("-" * 50)

# --- Cell 5: Train Final Model (XGBoost + scale_pos_weight) ---
# Train our winning model on the full training set.

print("\n--- Starting Final Model Training (XGBoost) ---")
final_model = xgb.XGBClassifier(
    objective='binary:logistic',
    random_state=SEED,
    n_jobs=-1,
    eval_metric='logloss',
    scale_pos_weight=scale_pos_weight_value,  # Key parameter for imbalance
    **best_params_xgb
)

start_time = time.time()
final_model.fit(X_train, y_train)
print(f"Training complete in {time.time() - start_time:.2f}s")
print("-" * 50)

# --- Cell 6: Generate Predictions ---
# Generate probabilities and apply the fixed decision threshold (Threshold = 0.8).

print("\n--- Generating predictions ---")
# Get probabilities for class 1 (fraud)
y_prob_train = final_model.predict_proba(X_train)[:, 1]
y_prob_test = final_model.predict_proba(X_test)[:, 1]

# Apply decision threshold that yielded the best cost balance in experiments
FINAL_THRESHOLD = 0.80
print(f"Applied Decision Threshold: {FINAL_THRESHOLD}")

y_pred_train = (y_prob_train > FINAL_THRESHOLD).astype(int)
y_pred_test = (y_prob_test > FINAL_THRESHOLD).astype(int)
print("-" * 50)

# --- Cell 7: Business Context and Metric Functions ---
# Define business costs and helper functions to calculate all 22 metrics.

# Cost parameters
TOTAL_FRAUD_VALUE_EUR = 60000
TOTAL_FRAUD_TRANSACTIONS = 492
COST_MULTIPLIER_LOW = 3.27
EUR_TO_PLN = 4.23
AVG_FRAUD_VALUE = TOTAL_FRAUD_VALUE_EUR / TOTAL_FRAUD_TRANSACTIONS
COST_PER_FN_EUR = AVG_FRAUD_VALUE * COST_MULTIPLIER_LOW
COST_PER_FN = COST_PER_FN_EUR * EUR_TO_PLN

print(f"\nCost per single False Negative (FN): {COST_PER_FN:.2f} PLN")


def calculate_ece_mce(y_true, y_prob, n_bins=10):
    """Calculates calibration errors: Expected Calibration Error and Max Calibration Error"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    ece = 0.0
    mce = 0.0
    for i in range(n_bins):
        in_bin = (y_prob > bin_lowers[i]) & (y_prob <= bin_uppers[i])
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            confidence_in_bin = np.mean(y_prob[in_bin])
            bin_error = np.abs(accuracy_in_bin - confidence_in_bin)
            ece += prop_in_bin * bin_error
            if bin_error > mce:
                mce = bin_error
    return ece, mce


def calculate_all_metrics(y_true, y_pred, y_prob):
    """Calculates a comprehensive set of metrics for the model."""
    metrics = {}
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics['TP'] = tp
    metrics['FP'] = fp
    metrics['TN'] = tn
    metrics['FN'] = fn
    metrics['Precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['Recall (TPR)'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['Specificity (TNR)'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics['FPR'] = fp / (fp + tn) if (fp + tn) > 0 else 0
    metrics['FNR'] = fn / (fn + tp) if (fn + tp) > 0 else 0
    metrics['NPV'] = tn / (tn + fn) if (tn + fn) > 0 else 0
    metrics['Error Rate'] = 1 - accuracy_score(y_true, y_pred)
    metrics['Accuracy'] = accuracy_score(y_true, y_pred)
    metrics['Balanced Acc'] = balanced_accuracy_score(y_true, y_pred)
    metrics['MCC'] = matthews_corrcoef(y_true, y_pred)
    metrics['F1 Score'] = f1_score(y_true, y_pred, zero_division=0)
    metrics['G-mean'] = np.sqrt(metrics['Recall (TPR)'] * metrics['Specificity (TNR)'])

    if len(np.unique(y_true)) > 1:
        metrics['AUC ROC'] = roc_auc_score(y_true, y_prob)
        pr_precision, pr_recall, _ = precision_recall_curve(y_true, y_prob)
        metrics['AUC PR'] = auc(pr_recall, pr_precision)
    else:
        metrics['AUC ROC'] = 0.5
        metrics['AUC PR'] = 0.0

    metrics['Brier Score'] = brier_score_loss(y_true, y_prob)
    metrics['Expected Cost'] = fn * COST_PER_FN  # Business cost
    metrics['ECE'], metrics['MCE'] = calculate_ece_mce(y_true.to_numpy(), y_prob)
    return metrics


# --- Cell 8: Calculate and Display Results ---
# Calculate metrics for both training and test sets.

metrics_train = calculate_all_metrics(y_train, y_pred_train, y_prob_train)
metrics_test = calculate_all_metrics(y_test, y_pred_test, y_prob_test)


# Create summary table
def create_summary_table(metric_list, train_metrics, test_metrics):
    data = {'Metric': metric_list, 'Train Set': [], 'Test Set': []}
    for metric in metric_list:
        val_train = train_metrics.get(metric)
        val_test = test_metrics.get(metric)

        if metric == 'Expected Cost':
            data['Train Set'].append(f"{val_train:,.2f} PLN")
            data['Test Set'].append(f"{val_test:,.2f} PLN")
        else:
            data['Train Set'].append(round(val_train, 4) if isinstance(val_train, (int, float)) else val_train)
            data['Test Set'].append(round(val_test, 4) if isinstance(val_test, (int, float)) else val_test)

    return pd.DataFrame(data).set_index('Metric')

# UPDATED LISTS TO INCLUDE MISSING METRICS
metrics_table_1 = [
    'TP', 'FP', 'TN', 'FN',
    'Precision', 'Recall (TPR)', 'Specificity (TNR)',
    'FPR', 'FNR', 'NPV', 'Error Rate'
]

metrics_table_2 = [
    'Accuracy', 'Balanced Acc', 'MCC', 'F1 Score', 'G-mean',
    'AUC PR', 'AUC ROC', 'Brier Score', 'ECE', 'MCE', 'Expected Cost'
]

print(f"\n{'=' * 80}")
print(f"FINAL MODEL RESULTS (XGBoost + scale_pos_weight)")
print(f"{'=' * 80}")

print("\n--- Table 1: Classification Metrics (Left Column) ---")
print(create_summary_table(metrics_table_1, metrics_train, metrics_test))

print("\n--- Table 2: Quality & Calibration Metrics (Right Column) ---")
print(create_summary_table(metrics_table_2, metrics_train, metrics_test))

# --- Cell 9: Visualize Confusion Matrix ---
# Visualize results on the test set.

plt.figure(figsize=(8, 6))
cm_test = confusion_matrix(y_test, y_pred_test)
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Legitimate', 'Predicted Fraud'],
            yticklabels=['Actual Legitimate', 'Actual Fraud'])
plt.title(f'Confusion Matrix - Final Model\n(Threshold={FINAL_THRESHOLD})')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

print("\nSummary:")
print(f"Model successfully evaluated. Total expected cost on test set: {metrics_test['Expected Cost']:,.2f} PLN.")
