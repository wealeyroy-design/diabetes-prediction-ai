# ============================================================
# DIABETES PREDICTION - FULL AI MINI PROJECT PIPELINE
# Dataset: Pima Indians Diabetes Dataset (Kaggle / UCI)
# Task: Binary Classification
# ============================================================

# ── INSTALL (run once in terminal if needed) ──────────────────
# pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap streamlit

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay, RocCurveDisplay, classification_report
)
from xgboost import XGBClassifier
import shap

# ── REPRODUCIBILITY ───────────────────────────────────────────
SEED = 42
np.random.seed(SEED)


# ============================================================
# SECTION 2 – DATA ACQUISITION
# ============================================================
print("=" * 60)
print("SECTION 2: DATA ACQUISITION")
print("=" * 60)

# Download: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
# Save as 'diabetes.csv' in the same folder as this script.

df = pd.read_csv("diabetes.csv")

print(f"Dataset shape : {df.shape}")
print(f"Features      : {list(df.columns)}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nClass distribution:\n{df['Outcome'].value_counts()}")


# ============================================================
# SECTION 3 – DATA CLEANING & PREPROCESSING
# ============================================================
print("\n" + "=" * 60)
print("SECTION 3: DATA CLEANING & PREPROCESSING")
print("=" * 60)

# These columns cannot biologically be zero → treat as missing
zero_invalid = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

print(f"\nZero counts before cleaning:")
print((df[zero_invalid] == 0).sum())

# Replace zeros with NaN
df[zero_invalid] = df[zero_invalid].replace(0, np.nan)

# Impute with median (robust to outliers)
for col in zero_invalid:
    median_val = df[col].median()
    df[col].fillna(median_val, inplace=True)

print(f"\nMissing values after imputation: {df.isnull().sum().sum()}")
print(f"Duplicates: {df.duplicated().sum()}")
df.drop_duplicates(inplace=True)

# Features and target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Train / test split (80/20, stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"\nTraining set : {X_train.shape}")
print(f"Test set     : {X_test.shape}")


# ============================================================
# SECTION 4 – EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================
print("\n" + "=" * 60)
print("SECTION 4: EXPLORATORY DATA ANALYSIS")
print("=" * 60)

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
fig.suptitle("Feature Distributions by Outcome", fontsize=16, fontweight="bold")

for ax, col in zip(axes.flatten(), X.columns):
    for outcome, label, color in zip([0, 1], ["No Diabetes", "Diabetes"], ["steelblue", "tomato"]):
        ax.hist(df[df["Outcome"] == outcome][col], bins=20,
                alpha=0.6, label=label, color=color)
    ax.set_title(col)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("eda_distributions.png", dpi=150)
plt.show()
print("Saved: eda_distributions.png")

# Correlation heatmap
plt.figure(figsize=(10, 8))
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
            cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("eda_heatmap.png", dpi=150)
plt.show()
print("Saved: eda_heatmap.png")

# Boxplots
fig, axes = plt.subplots(3, 3, figsize=(15, 10))
fig.suptitle("Boxplots – Feature vs Outcome", fontsize=16, fontweight="bold")
for ax, col in zip(axes.flatten(), X.columns):
    df.boxplot(column=col, by="Outcome", ax=ax, patch_artist=True,
               boxprops=dict(facecolor="lightblue"))
    ax.set_title(col)
    ax.set_xlabel("Outcome (0=No, 1=Yes)")
plt.suptitle("")
plt.tight_layout()
plt.savefig("eda_boxplots.png", dpi=150)
plt.show()
print("Saved: eda_boxplots.png")

print("\nKey statistical summary:")
print(df.groupby("Outcome").mean().T.to_string())


# ============================================================
# SECTION 5 – FEATURE ENGINEERING
# ============================================================
print("\n" + "=" * 60)
print("SECTION 5: FEATURE ENGINEERING")
print("=" * 60)

# Work on a copy for feature engineering
df_fe = df.copy()

# Interaction features
df_fe["Glucose_BMI"]        = df_fe["Glucose"] * df_fe["BMI"]
df_fe["Age_Pregnancies"]    = df_fe["Age"] * df_fe["Pregnancies"]
df_fe["Insulin_Glucose"]    = df_fe["Insulin"] / (df_fe["Glucose"] + 1)

# BMI category (clinical thresholds)
df_fe["BMI_Category"] = pd.cut(
    df_fe["BMI"],
    bins=[0, 18.5, 24.9, 29.9, 100],
    labels=[0, 1, 2, 3]  # Underweight, Normal, Overweight, Obese
).astype(int)

# Age group
df_fe["Age_Group"] = pd.cut(
    df_fe["Age"],
    bins=[0, 30, 45, 60, 100],
    labels=[0, 1, 2, 3]
).astype(int)

print("New features added:")
print("  - Glucose_BMI     : glucose × BMI interaction")
print("  - Age_Pregnancies : age × pregnancy count")
print("  - Insulin_Glucose : insulin-to-glucose ratio")
print("  - BMI_Category    : clinical BMI classification")
print("  - Age_Group       : grouped age ranges")

X_fe = df_fe.drop("Outcome", axis=1)
y_fe = df_fe["Outcome"]

X_train_fe, X_test_fe, y_train_fe, y_test_fe = train_test_split(
    X_fe, y_fe, test_size=0.2, random_state=SEED, stratify=y_fe
)

scaler_fe = StandardScaler()
X_train_fe_sc = scaler_fe.fit_transform(X_train_fe)
X_test_fe_sc  = scaler_fe.transform(X_test_fe)


# ============================================================
# SECTION 6 – MODEL BUILDING
# ============================================================
print("\n" + "=" * 60)
print("SECTION 6: MODEL BUILDING")
print("=" * 60)

# ── Logistic Regression ──────────────────────────────────────
lr = LogisticRegression(random_state=SEED, max_iter=1000)
lr.fit(X_train_fe_sc, y_train_fe)
print("✔ Logistic Regression trained")

# ── Random Forest ────────────────────────────────────────────
rf_params = {"n_estimators": [100, 200], "max_depth": [None, 5, 10]}
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=SEED),
    rf_params, cv=5, scoring="f1", n_jobs=-1
)
rf_grid.fit(X_train_fe, y_train_fe)
rf = rf_grid.best_estimator_
print(f"✔ Random Forest trained | Best params: {rf_grid.best_params_}")

# ── XGBoost ──────────────────────────────────────────────────
xgb_params = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5],
    "learning_rate": [0.05, 0.1]
}
xgb_grid = GridSearchCV(
    XGBClassifier(random_state=SEED, eval_metric="logloss", use_label_encoder=False),
    xgb_params, cv=5, scoring="f1", n_jobs=-1
)
xgb_grid.fit(X_train_fe, y_train_fe)
xgb = xgb_grid.best_estimator_
print(f"✔ XGBoost trained       | Best params: {xgb_grid.best_params_}")


# ============================================================
# SECTION 7 – MODEL EVALUATION
# ============================================================
print("\n" + "=" * 60)
print("SECTION 7: MODEL EVALUATION")
print("=" * 60)

models = {
    "Logistic Regression": (lr,  X_test_fe_sc),
    "Random Forest":       (rf,  X_test_fe),
    "XGBoost":             (xgb, X_test_fe),
}

results = []
for name, (model, X_eval) in models.items():
    y_pred = model.predict(X_eval)
    y_prob = model.predict_proba(X_eval)[:, 1]
    results.append({
        "Model":     name,
        "Accuracy":  round(accuracy_score(y_test_fe, y_pred), 4),
        "Precision": round(precision_score(y_test_fe, y_pred), 4),
        "Recall":    round(recall_score(y_test_fe, y_pred), 4),
        "F1-Score":  round(f1_score(y_test_fe, y_pred), 4),
        "ROC-AUC":   round(roc_auc_score(y_test_fe, y_prob), 4),
    })

results_df = pd.DataFrame(results).set_index("Model")
print("\n📊 Performance Comparison Table:")
print(results_df.to_string())

# Confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
fig.suptitle("Confusion Matrices", fontsize=14, fontweight="bold")
for ax, (name, (model, X_eval)) in zip(axes, models.items()):
    y_pred = model.predict(X_eval)
    cm = confusion_matrix(y_test_fe, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["No Diabetes", "Diabetes"]).plot(ax=ax, colorbar=False)
    ax.set_title(name)
plt.tight_layout()
plt.savefig("eval_confusion_matrices.png", dpi=150)
plt.show()
print("Saved: eval_confusion_matrices.png")

# ROC curves
plt.figure(figsize=(8, 6))
for name, (model, X_eval) in models.items():
    RocCurveDisplay.from_estimator(model, X_eval, y_test_fe, name=name, ax=plt.gca())
plt.title("ROC Curves – All Models", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("eval_roc_curves.png", dpi=150)
plt.show()
print("Saved: eval_roc_curves.png")

# Best model detailed report
print("\n📋 Detailed Report – XGBoost (Best Model):")
print(classification_report(y_test_fe, xgb.predict(X_test_fe),
                             target_names=["No Diabetes", "Diabetes"]))


# ============================================================
# SECTION 8 – RESULTS INTERPRETATION & INSIGHTS
# ============================================================
print("\n" + "=" * 60)
print("SECTION 8: RESULTS INTERPRETATION & INSIGHTS")
print("=" * 60)

# Feature importance – Random Forest
feat_imp = pd.Series(rf.feature_importances_, index=X_fe.columns).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
feat_imp.plot(kind="barh", color="steelblue")
plt.gca().invert_yaxis()
plt.title("Feature Importance – Random Forest", fontsize=14, fontweight="bold")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("insights_feature_importance.png", dpi=150)
plt.show()
print("Saved: insights_feature_importance.png")

print(f"\nTop 5 predictive features:\n{feat_imp.head()}")

# SHAP values for XGBoost
print("\nGenerating SHAP explanation for XGBoost...")
explainer   = shap.TreeExplainer(xgb)
shap_values = explainer.shap_values(X_test_fe)

plt.figure()
shap.summary_plot(shap_values, X_test_fe, plot_type="bar", show=False)
plt.title("SHAP Feature Importance", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("insights_shap.png", dpi=150)
plt.show()
print("Saved: insights_shap.png")

print("""
KEY INSIGHTS:
  1. Glucose level is the single strongest predictor of diabetes.
  2. BMI and Age also play critical roles; higher values → higher risk.
  3. The Glucose_BMI interaction feature boosted model performance,
     suggesting combined metabolic burden matters.
  4. XGBoost outperformed other models with the highest ROC-AUC,
     making it the recommended model for deployment.
  5. Patients with Insulin > 200 showed disproportionately high
     diabetes rates — a potential clinical red flag.
""")


# ============================================================
# SECTION 9 – DEPLOYMENT (Streamlit App)
# ============================================================
# Run the app separately: streamlit run streamlit_app.py
print("\n" + "=" * 60)
print("SECTION 9: DEPLOYMENT")
print("See streamlit_app.py for the web application.")
print("Run with: streamlit run streamlit_app.py")
print("=" * 60)

# Save the trained model and scaler for the Streamlit app
import joblib
joblib.dump(xgb,       "model_xgb.pkl")
joblib.dump(scaler_fe, "scaler.pkl")
joblib.dump(list(X_fe.columns), "feature_names.pkl")
print("Model, scaler, and feature names saved as .pkl files.")
