from sklearn.datasets import fetch_california_housing
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats

from sklearn.model_selection import train_test_split, RepeatedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import (
    LinearRegression,
    RidgeCV,
    LassoCV,
    ElasticNetCV,
    HuberRegressor
)
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
    explained_variance_score
)
from scipy.stats import ttest_rel

# =========================================================
# 0) GENERAL STYLE SETTINGS
# =========================================================
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 11
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["legend.fontsize"] = 10

# =========================================================
# 1) LOAD DATA
# =========================================================
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["Price"] = data.target

print("First 5 rows:")
print(df.head())

print("\nDataset info:")
print(df.info())

print("\nDescriptive statistics:")
print(df.describe())

# =========================================================
# 2) FIGURE 1 - CORRELATION HEATMAP
# =========================================================
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    df.corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    cbar=True,
    square=False,
    linewidths=0.5,
    ax=ax
)
ax.set_title("Figure 1. Correlation Heatmap of the California Housing Variables", pad=12)
plt.tight_layout()
plt.savefig("figure1_correlation_heatmap.png", bbox_inches="tight")
plt.show()

# =========================================================
# 3) FIGURE 2 - FEATURE DISTRIBUTIONS
# =========================================================
# =========================================================
# 3) FIGURE 2 - FEATURE DISTRIBUTIONS
# =========================================================
fig, axes = plt.subplots(3, 3, figsize=(13, 10))

columns = df.columns.tolist()

for i, ax in enumerate(axes.flat):
    col = columns[i]
    ax.hist(df[col], bins=30, edgecolor="black")
    ax.set_title(col, fontsize=12, pad=10)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    # Yazı kalabalığını azalt
    ax.tick_params(axis='both', labelsize=9)

    # Üst iki satırda x ekseni etiketlerini gizle
    if i < 6:
        ax.tick_params(axis='x', labelbottom=False)

fig.suptitle(
    "Figure 2. Distribution of Input Features and Target Variable",
    fontsize=16,
    y=0.98
)


plt.subplots_adjust(top=0.90, bottom=0.08, left=0.07, right=0.98, hspace=0.30, wspace=0.18)

plt.savefig("figure2_feature_distributions.png", dpi=300, bbox_inches="tight")
plt.show()

# =========================================================
# 4) PREPARE TRAIN / TEST DATA
# =========================================================
X = df.drop("Price", axis=1)
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================================================
# 5) DEFINE MODELS
# =========================================================
linear_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])

ridge_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0]))
])

lasso_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LassoCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0], max_iter=10000))
])

elastic_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", ElasticNetCV(
        l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
        alphas=[0.001, 0.01, 0.1, 1.0, 10.0],
        max_iter=10000
    ))
])

huber_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", HuberRegressor())
])

gbr_model = GradientBoostingRegressor(random_state=42)

models = {
    "Linear Regression": linear_model,
    "Ridge": ridge_model,
    "Lasso": lasso_model,
    "Elastic Net": elastic_model,
    "Gradient Boosting": gbr_model,
    "Huber": huber_model
}

# =========================================================
# 6) METRIC FUNCTION
# =========================================================
def adjusted_r2(r2_value, n, p):
    return 1 - (1 - r2_value) * (n - 1) / (n - p - 1)

# =========================================================
# 7) SINGLE TEST-SPLIT RESULTS
# =========================================================
results_list = []
n = X_test.shape[0]
p = X_test.shape[1]

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    results_list.append({
        "Model": name,
        "MAE": mean_absolute_error(y_test, pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
        "R2": r2_score(y_test, pred),
        "Adjusted_R2": adjusted_r2(r2_score(y_test, pred), n, p),
        "MAPE": mean_absolute_percentage_error(y_test, pred),
        "Explained_Var": explained_variance_score(y_test, pred)
    })

results = pd.DataFrame(results_list)

print("\nSingle split results:")
print(results)

# =========================================================
# 8) CROSS-VALIDATION RESULTS
# =========================================================
cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)

scoring = {
    "mae": "neg_mean_absolute_error",
    "rmse": "neg_root_mean_squared_error",
    "r2": "r2"
}

cv_results = []

for name, model in models.items():
    scores = cross_validate(model, X, y, cv=cv, scoring=scoring)

    cv_results.append({
        "Model": name,
        "MAE_mean": -scores["test_mae"].mean(),
        "MAE_std": scores["test_mae"].std(),
        "RMSE_mean": -scores["test_rmse"].mean(),
        "RMSE_std": scores["test_rmse"].std(),
        "R2_mean": scores["test_r2"].mean(),
        "R2_std": scores["test_r2"].std()
    })

cv_df = pd.DataFrame(cv_results)

print("\nCross-validation results:")
print(cv_df)

# =========================================================
# 9) BEST MODEL RESIDUAL ANALYSIS
# =========================================================
gbr_model.fit(X_train, y_train)
gbr_pred = gbr_model.predict(X_test)
residuals = y_test - gbr_pred

# Figure 3 - Residual plot
fig, ax = plt.subplots(figsize=(8.5, 5.5))
ax.scatter(gbr_pred, residuals, alpha=0.45, edgecolors="none")
ax.axhline(y=0, color="red", linestyle="--", linewidth=1.5)
ax.set_xlabel("Predicted Values")
ax.set_ylabel("Residuals")
ax.set_title("Figure 3. Residual Plot for the Gradient Boosting Regressor", pad=12)
ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
plt.tight_layout()
plt.savefig("figure3_residual_plot.png", bbox_inches="tight")
plt.show()

# Figure 4 - Q-Q plot
fig = plt.figure(figsize=(8.5, 5.5))
ax = fig.add_subplot(111)
stats.probplot(residuals, dist="norm", plot=ax)
ax.set_title("Figure 4. Q-Q Plot of Gradient Boosting Residuals", pad=12)
ax.grid(False)
plt.tight_layout()
plt.savefig("figure4_qq_plot.png", bbox_inches="tight")
plt.show()

# =========================================================
# 10) FEATURE IMPORTANCE
# =========================================================
importances = pd.DataFrame({
    "Feature": X.columns,
    "Importance": gbr_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nFeature importances:")
print(importances)

# Figure 5 - Feature importance bar chart
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=importances, x="Importance", y="Feature", ax=ax)
ax.set_title("Figure 5. Feature Importance Scores from the Gradient Boosting Regressor", pad=12)
ax.set_xlabel("Importance Score")
ax.set_ylabel("Feature")
ax.grid(True, axis="x", linestyle="--", linewidth=0.5, alpha=0.5)
plt.tight_layout()
plt.savefig("figure5_feature_importance.png", bbox_inches="tight")
plt.show()

# =========================================================
# 11) TOP-5 FEATURE EXPERIMENT
# =========================================================
top_features = importances["Feature"].head(5).tolist()

X_top = df[top_features]
y_top = df["Price"]

X_train_top, X_test_top, y_train_top, y_test_top = train_test_split(
    X_top, y_top, test_size=0.2, random_state=42
)

gbr_top = GradientBoostingRegressor(random_state=42)
gbr_top.fit(X_train_top, y_train_top)
gbr_top_pred = gbr_top.predict(X_test_top)

top5_rmse = np.sqrt(mean_squared_error(y_test_top, gbr_top_pred))
top5_r2 = r2_score(y_test_top, gbr_top_pred)

print("\nTop-5 feature GBR results:")
print("Top-5 GBR RMSE:", top5_rmse)
print("Top-5 GBR R2:", top5_r2)

# =========================================================
# 12) POLYNOMIAL FEATURE EXPERIMENT
# =========================================================
poly_model = Pipeline([
    ("scaler", StandardScaler()),
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("model", LinearRegression())
])

poly_model.fit(X_train, y_train)
poly_pred = poly_model.predict(X_test)

poly_rmse = np.sqrt(mean_squared_error(y_test, poly_pred))
poly_r2 = r2_score(y_test, poly_pred)

print("\nPolynomial linear results:")
print("Polynomial Linear RMSE:", poly_rmse)
print("Polynomial Linear R2:", poly_r2)

# =========================================================
# 13) LOG-TARGET EXPERIMENT
# =========================================================
y_log = np.log1p(df["Price"])

X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(
    X, y_log, test_size=0.2, random_state=42
)

gbr_log = GradientBoostingRegressor(random_state=42)
gbr_log.fit(X_train_log, y_train_log)
gbr_log_pred = gbr_log.predict(X_test_log)

log_rmse = np.sqrt(mean_squared_error(y_test_log, gbr_log_pred))
log_r2 = r2_score(y_test_log, gbr_log_pred)

gbr_log_pred_original = np.expm1(gbr_log_pred)
y_test_log_original = np.expm1(y_test_log)

log_original_rmse = np.sqrt(mean_squared_error(y_test_log_original, gbr_log_pred_original))
log_original_r2 = r2_score(y_test_log_original, gbr_log_pred_original)

print("\nLog-target GBR results:")
print("Log-Target GBR RMSE:", log_rmse)
print("Log-Target GBR R2:", log_r2)
print("Log-Target GBR Original RMSE:", log_original_rmse)
print("Log-Target GBR Original R2:", log_original_r2)

# =========================================================
# 14) PAIRED T-TEST: GBR VS LASSO
# =========================================================
lasso_model.fit(X_train, y_train)
lasso_pred = lasso_model.predict(X_test)

gbr_errors = np.abs(y_test - gbr_pred)
lasso_errors = np.abs(y_test - lasso_pred)

t_stat, p_value = ttest_rel(lasso_errors, gbr_errors)

print("\nPaired t-test results:")
print("Paired t-test t-statistic:", t_stat)
print("Paired t-test p-value:", p_value)

# =========================================================
# 15) INTERACTION TERM EXPERIMENT
# =========================================================
df_inter = df.copy()
df_inter["MedInc_AveRooms"] = df_inter["MedInc"] * df_inter["AveRooms"]

X_inter = df_inter.drop("Price", axis=1)
y_inter = df_inter["Price"]

X_train_inter, X_test_inter, y_train_inter, y_test_inter = train_test_split(
    X_inter, y_inter, test_size=0.2, random_state=42
)

linear_inter = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])

linear_inter.fit(X_train_inter, y_train_inter)
inter_pred = linear_inter.predict(X_test_inter)

inter_rmse = np.sqrt(mean_squared_error(y_test_inter, inter_pred))
inter_r2 = r2_score(y_test_inter, inter_pred)

print("\nInteraction term results:")
print("Interaction Linear RMSE:", inter_rmse)
print("Interaction Linear R2:", inter_r2)

# =========================================================
# 16) FINAL SUMMARY TABLE DATA
# =========================================================
summary_results = pd.DataFrame([
    ["Linear Regression", 0.7455813830127763, 0.575787706032451],
    ["Ridge", 0.7450104387795742, 0.5764371557987773],
    ["Lasso", 0.7442405630689863, 0.5773121026225017],
    ["Elastic Net", 0.7446201774067472, 0.5768807923267094],
    ["Gradient Boosting", 0.5422152016168362, 0.7756446042829697],
    ["Huber", 0.7584447344094588, 0.5610237531143614],
    ["Top-5 Feature GBR", 0.5406306417230388, 0.7769539925144401],
    ["Polynomial Linear", 0.6813967448044601, 0.6456819729261963],
    ["Log-Target GBR (original scale)", 0.544134535459832, 0.7740534462168064],
    ["Interaction Linear", 0.7420870632181686, 0.5797547036536488]
], columns=["Model", "RMSE", "R2"])

final_summary = summary_results.sort_values(by="RMSE").reset_index(drop=True)
final_summary.index = final_summary.index + 1
final_summary.index.name = "Rank"

print("\nFinal ranked summary:")
print(final_summary)

# =========================================================
# 17) TABLE 1 - MAIN RESULTS TABLE
# =========================================================
table1 = final_summary.round(4)

fig, ax = plt.subplots(figsize=(11, 4.8))
ax.axis("off")

tbl = ax.table(
    cellText=table1.values,
    colLabels=table1.columns,
    rowLabels=table1.index,
    loc="center",
    cellLoc="center"
)

tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1.15, 1.5)

ax.set_title("Table 1. Test-Set Performance Comparison of Regression Models", pad=14)
plt.tight_layout()
plt.savefig("table1_test_set_results.png", bbox_inches="tight")
plt.show()

# =========================================================
# 18) TABLE 2 - CROSS-VALIDATION TABLE
# =========================================================
cv_table = cv_df.copy().round(4)

fig, ax = plt.subplots(figsize=(12, 3.8))
ax.axis("off")

tbl = ax.table(
    cellText=cv_table.values,
    colLabels=cv_table.columns,
    loc="center",
    cellLoc="center"
)

tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1.12, 1.45)

ax.set_title("Table 2. Repeated 5-Fold Cross-Validation Results", pad=14)
plt.tight_layout()
plt.savefig("table2_cross_validation_results.png", bbox_inches="tight")
plt.show()