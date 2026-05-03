import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    precision_recall_curve,
    PrecisionRecallDisplay,
    make_scorer
)

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler

from xgboost import XGBClassifier

# =========================================================
# 0) GENERAL STYLE
# =========================================================
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 11

# =========================================================
# 1) LOAD DATA
# =========================================================
data = fetch_openml(name="creditcard", version=1, as_frame=True)
df = data.frame

print("First 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nClass distribution:")
print(df["Class"].value_counts())

# =========================================================
# 2) IMBALANCE RATIO
# =========================================================
counts = df["Class"].value_counts()
majority = counts.iloc[0]
minority = counts.iloc[1]
ir = majority / minority

print("\nImbalance Ratio:", ir)

# =========================================================
# 3) MISSING VALUES
# =========================================================
print("\nMissing values:")
print(df.isnull().sum())

# =========================================================
# 4) TRAIN-TEST SPLIT
# =========================================================
X = df.drop("Class", axis=1)
y = df["Class"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nX_train:", X_train.shape)
print("X_test:", X_test.shape)

print("\ny_train class counts:")
print(y_train.value_counts())

print("\ny_test class counts:")
print(y_test.value_counts())

# =========================================================
# 5) RESULT STORAGE
# =========================================================
results = []
threshold_results = []

# =========================================================
# 6) GENERAL EVALUATION FUNCTION
# =========================================================
def evaluate_model(model_name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"\n{'='*70}")
    print(model_name.upper())
    print(f"{'='*70}")

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1_binary = f1_score(y_test, y_pred, zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_micro = f1_score(y_test, y_pred, average="micro", zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    brier = brier_score_loss(y_test, y_prob)

    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 (binary):", f1_binary)
    print("F1 (macro):", f1_macro)
    print("F1 (micro):", f1_micro)
    print("ROC-AUC:", roc_auc)
    print("PR-AUC:", pr_auc)
    print("Balanced Accuracy:", balanced_acc)
    print("MCC:", mcc)
    print("Brier Score:", brier)

    results.append({
        "Model": model_name,
        "Precision": precision,
        "Recall": recall,
        "F1": f1_binary,
        "F1 Macro": f1_macro,
        "F1 Micro": f1_micro,
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc,
        "Balanced Accuracy": balanced_acc,
        "MCC": mcc,
        "Brier Score": brier
    })

    return y_pred, y_prob

# =========================================================
# 7) THRESHOLD OPTIMIZATION FUNCTION
# =========================================================
def find_best_threshold(y_true, y_prob, model_name):
    thresholds = np.arange(0.01, 1.00, 0.01)

    best_threshold = 0.50
    best_precision = 0
    best_recall = 0
    best_f1 = -1

    for threshold in thresholds:
        y_pred_thresh = (y_prob >= threshold).astype(int)

        precision = precision_score(y_true, y_pred_thresh, zero_division=0)
        recall = recall_score(y_true, y_pred_thresh, zero_division=0)
        f1 = f1_score(y_true, y_pred_thresh, zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_precision = precision
            best_recall = recall

    print(f"\n{'='*70}")
    print(f"{model_name.upper()} - OPTIMAL THRESHOLD")
    print(f"{'='*70}")
    print("Best Threshold:", best_threshold)
    print("Best Precision:", best_precision)
    print("Best Recall:", best_recall)
    print("Best F1:", best_f1)

    final_pred = (y_prob >= best_threshold).astype(int)

    print("\nConfusion Matrix at Best Threshold:")
    print(confusion_matrix(y_true, final_pred))

    print("\nClassification Report at Best Threshold:")
    print(classification_report(y_true, final_pred, digits=4))

    threshold_results.append({
        "Model": model_name,
        "Best Threshold": best_threshold,
        "Precision": best_precision,
        "Recall": best_recall,
        "Best F1": best_f1
    })

# =========================================================
# 8) DEFINE MODELS
# =========================================================
log_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000, random_state=42))
])

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    random_state=42
)

classes = np.array([0, 1])
weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
class_weight_dict = {0: weights[0], 1: weights[1]}
sample_weights = y_train.map(class_weight_dict)

mlp_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=150,
        random_state=42
    ))
])

smote_log_model = ImbPipeline([
    ("scaler", StandardScaler()),
    ("smote", SMOTE(random_state=42)),
    ("model", LogisticRegression(max_iter=1000, random_state=42))
])

adasyn_log_model = ImbPipeline([
    ("scaler", StandardScaler()),
    ("adasyn", ADASYN(random_state=42)),
    ("model", LogisticRegression(max_iter=1000, random_state=42))
])

under_log_model = ImbPipeline([
    ("scaler", StandardScaler()),
    ("undersample", RandomUnderSampler(random_state=42)),
    ("model", LogisticRegression(max_iter=1000, random_state=42))
])

rf_calibrated = CalibratedClassifierCV(
    estimator=RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),
    method="sigmoid",
    cv=3
)

xgb_calibrated = CalibratedClassifierCV(
    estimator=XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42
    ),
    method="sigmoid",
    cv=3
)

# =========================================================
# 9) EVALUATE STANDARD MODELS
# =========================================================
_, log_prob = evaluate_model("Logistic Regression", log_model, X_train, X_test, y_train, y_test)
_, rf_prob = evaluate_model("Random Forest", rf_model, X_train, X_test, y_train, y_test)
_, xgb_prob = evaluate_model("XGBoost", xgb_model, X_train, X_test, y_train, y_test)

# MLP needs sample weights separately
mlp_model.fit(X_train, y_train, model__sample_weight=sample_weights)
mlp_pred = mlp_model.predict(X_test)
mlp_prob = mlp_model.predict_proba(X_test)[:, 1]

print(f"\n{'='*70}")
print("MLP")
print(f"{'='*70}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, mlp_pred))
print("\nClassification Report:")
print(classification_report(y_test, mlp_pred, digits=4))

mlp_precision = precision_score(y_test, mlp_pred, zero_division=0)
mlp_recall = recall_score(y_test, mlp_pred, zero_division=0)
mlp_f1 = f1_score(y_test, mlp_pred, zero_division=0)
mlp_f1_macro = f1_score(y_test, mlp_pred, average="macro", zero_division=0)
mlp_f1_micro = f1_score(y_test, mlp_pred, average="micro", zero_division=0)
mlp_roc_auc = roc_auc_score(y_test, mlp_prob)
mlp_pr_auc = average_precision_score(y_test, mlp_prob)
mlp_bal_acc = balanced_accuracy_score(y_test, mlp_pred)
mlp_mcc = matthews_corrcoef(y_test, mlp_pred)
mlp_brier = brier_score_loss(y_test, mlp_prob)

print("Precision:", mlp_precision)
print("Recall:", mlp_recall)
print("F1 (binary):", mlp_f1)
print("F1 (macro):", mlp_f1_macro)
print("F1 (micro):", mlp_f1_micro)
print("ROC-AUC:", mlp_roc_auc)
print("PR-AUC:", mlp_pr_auc)
print("Balanced Accuracy:", mlp_bal_acc)
print("MCC:", mlp_mcc)
print("Brier Score:", mlp_brier)

results.append({
    "Model": "MLP",
    "Precision": mlp_precision,
    "Recall": mlp_recall,
    "F1": mlp_f1,
    "F1 Macro": mlp_f1_macro,
    "F1 Micro": mlp_f1_micro,
    "ROC-AUC": mlp_roc_auc,
    "PR-AUC": mlp_pr_auc,
    "Balanced Accuracy": mlp_bal_acc,
    "MCC": mlp_mcc,
    "Brier Score": mlp_brier
})

# =========================================================
# 10) EVALUATE RESAMPLING MODELS
# =========================================================
evaluate_model("SMOTE + Logistic Regression", smote_log_model, X_train, X_test, y_train, y_test)
evaluate_model("ADASYN + Logistic Regression", adasyn_log_model, X_train, X_test, y_train, y_test)
evaluate_model("Random Undersampling + Logistic Regression", under_log_model, X_train, X_test, y_train, y_test)

# =========================================================
# 11) EVALUATE CALIBRATED MODELS
# =========================================================
_, rf_cal_prob = evaluate_model("Random Forest Calibrated", rf_calibrated, X_train, X_test, y_train, y_test)
_, xgb_cal_prob = evaluate_model("XGBoost Calibrated", xgb_calibrated, X_train, X_test, y_train, y_test)

# =========================================================
# 12) THRESHOLD OPTIMIZATION FOR BEST TWO MODELS
# =========================================================
find_best_threshold(y_test, rf_prob, "Random Forest")
find_best_threshold(y_test, xgb_prob, "XGBoost")

# =========================================================
# 13) PRECISION-RECALL CURVES
# =========================================================
rf_precision_curve, rf_recall_curve, _ = precision_recall_curve(y_test, rf_prob)
xgb_precision_curve, xgb_recall_curve, _ = precision_recall_curve(y_test, xgb_prob)

rf_ap = average_precision_score(y_test, rf_prob)
xgb_ap = average_precision_score(y_test, xgb_prob)

plt.figure(figsize=(8, 6))
plt.plot(
    rf_recall_curve,
    rf_precision_curve,
    linewidth=2.5,
    linestyle="-",
    label=f"Random Forest (AP = {rf_ap:.2f})"
)
plt.plot(
    xgb_recall_curve,
    xgb_precision_curve,
    linewidth=2.5,
    linestyle="--",
    label=f"XGBoost (AP = {xgb_ap:.2f})"
)
plt.title("Precision-Recall Curves for the Best Two Models")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
plt.legend(loc="lower left")
plt.tight_layout()
plt.savefig("q2_pr_curves.png", bbox_inches="tight")
plt.show()

# =========================================================
# 14) CALIBRATION CURVES - RANDOM FOREST
# =========================================================
fig, ax = plt.subplots(figsize=(8, 6))
CalibrationDisplay.from_predictions(
    y_test,
    rf_prob,
    n_bins=10,
    name="Random Forest",
    ax=ax
)
CalibrationDisplay.from_predictions(
    y_test,
    rf_cal_prob,
    n_bins=10,
    name="Random Forest Calibrated",
    ax=ax
)
ax.set_title("Calibration Curves for Random Forest Before and After Calibration")
ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
plt.tight_layout()
plt.savefig("q2_rf_calibration_curve.png", bbox_inches="tight")
plt.show()

# =========================================================
# 15) CALIBRATION CURVES - XGBOOST
# =========================================================
fig, ax = plt.subplots(figsize=(8, 6))
CalibrationDisplay.from_predictions(
    y_test,
    xgb_prob,
    n_bins=10,
    name="XGBoost",
    ax=ax
)
CalibrationDisplay.from_predictions(
    y_test,
    xgb_cal_prob,
    n_bins=10,
    name="XGBoost Calibrated",
    ax=ax
)
ax.set_title("Calibration Curves for XGBoost Before and After Calibration")
ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
plt.tight_layout()
plt.savefig("q2_xgb_calibration_curve.png", bbox_inches="tight")
plt.show()

# =========================================================
# 16) CROSS-VALIDATION (CORE MODELS + RESAMPLING VARIANTS)
# =========================================================
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

scoring = {
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "f1_macro": make_scorer(f1_score, average="macro", zero_division=0),
    "f1_micro": make_scorer(f1_score, average="micro", zero_division=0),
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "balanced_accuracy": "balanced_accuracy",
    "mcc": make_scorer(matthews_corrcoef)
}

cv_models = {
    "Logistic Regression": log_model,
    "Random Forest": rf_model,
    "XGBoost": xgb_model,
    "SMOTE + Logistic Regression": smote_log_model,
    "ADASYN + Logistic Regression": adasyn_log_model,
    "Random Undersampling + Logistic Regression": under_log_model
}

cv_results_list = []

print(f"\n{'='*70}")
print("CROSS-VALIDATION RESULTS")
print(f"{'='*70}")

for name, model in cv_models.items():
    scores = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)

    row = {
        "Model": name,
        "Precision CV Mean": scores["test_precision"].mean(),
        "Recall CV Mean": scores["test_recall"].mean(),
        "F1 CV Mean": scores["test_f1"].mean(),
        "F1 Macro CV Mean": scores["test_f1_macro"].mean(),
        "F1 Micro CV Mean": scores["test_f1_micro"].mean(),
        "ROC-AUC CV Mean": scores["test_roc_auc"].mean(),
        "PR-AUC CV Mean": scores["test_pr_auc"].mean(),
        "Balanced Accuracy CV Mean": scores["test_balanced_accuracy"].mean(),
        "MCC CV Mean": scores["test_mcc"].mean()
    }
    cv_results_list.append(row)

cv_results_df = pd.DataFrame(cv_results_list)
print(cv_results_df)

# =========================================================
# 17) FINAL COMPARISON TABLES
# =========================================================
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="PR-AUC", ascending=False).reset_index(drop=True)

threshold_df = pd.DataFrame(threshold_results)
threshold_df = threshold_df.sort_values(by="Best F1", ascending=False).reset_index(drop=True)

print(f"\n{'='*70}")
print("FINAL MODEL COMPARISON")
print(f"{'='*70}")
print(results_df)

print(f"\n{'='*70}")
print("THRESHOLD OPTIMIZATION SUMMARY")
print(f"{'='*70}")
print(threshold_df)