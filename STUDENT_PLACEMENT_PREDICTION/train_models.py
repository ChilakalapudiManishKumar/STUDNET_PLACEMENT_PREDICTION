"""
Student Placement Prediction - Train, Evaluate, Compare Models, Save Best

FOLDER SETUP:
    your_project_folder/
        train_models.py                    <- this file
        X_train.csv, X_test.csv, y_train.csv, y_test.csv   <- created by prep_model_data.py

Creates in the same folder:
        model_comparison.csv
        best_model.joblib
        plots/10_confusion_matrix_best.png
        plots/11_model_comparison.png
        plots/12_feature_importance.png
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              roc_auc_score, confusion_matrix, classification_report)

# XGBoost is optional per the original spec — skip gracefully if not installed
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("xgboost not installed — skipping it. Install with: pip install xgboost")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOTS, exist_ok=True)

required_files = ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]
for f in required_files:
    if not os.path.exists(os.path.join(BASE_DIR, f)):
        raise FileNotFoundError(
            f"Could not find {f} in {BASE_DIR}\n"
            "Run prep_model_data.py first — it creates the train/test split files."
        )

X_train = pd.read_csv(os.path.join(BASE_DIR, "X_train.csv"))
X_test = pd.read_csv(os.path.join(BASE_DIR, "X_test.csv"))
y_train = pd.read_csv(os.path.join(BASE_DIR, "y_train.csv")).squeeze()
y_test = pd.read_csv(os.path.join(BASE_DIR, "y_test.csv")).squeeze()

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42),
    "SVM": SVC(kernel="rbf", probability=True, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=9),
}
if HAS_XGB:
    models["XGBoost"] = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                       eval_metric="logloss", random_state=42)

results = []
fitted = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    fitted[name] = model
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, proba)

    results.append({"Model": name, "Accuracy": acc, "Precision": prec,
                     "Recall": rec, "F1": f1, "ROC_AUC": auc})

    print(f"\n=== {name} ===")
    print(classification_report(y_test, preds, target_names=["No", "Yes"]))

results_df = pd.DataFrame(results).sort_values("F1", ascending=False).reset_index(drop=True)
print("\n=== MODEL COMPARISON TABLE ===")
print(results_df.round(4).to_string(index=False))

results_df.to_csv(os.path.join(BASE_DIR, "model_comparison.csv"), index=False)

best_name = results_df.iloc[0]["Model"]
best_model = fitted[best_name]
print(f"\nBest model by F1 score: {best_name}")

# Confusion matrix for best model
best_preds = best_model.predict(X_test)
cm = confusion_matrix(y_test, best_preds)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix - {best_name}")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "10_confusion_matrix_best.png"), dpi=110)
plt.close()

# Model comparison bar chart
plt.figure(figsize=(9, 5))
melt = results_df.melt(id_vars="Model", value_vars=["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"])
sns.barplot(data=melt, x="Model", y="value", hue="variable")
plt.xticks(rotation=20)
plt.ylim(0, 1)
plt.title("Model Comparison Across Metrics")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "11_model_comparison.png"), dpi=110)
plt.close()

# Feature importance (Random Forest)
importance_source = fitted["Random Forest"]
importances = importance_source.feature_importances_
feat_imp = pd.Series(importances, index=X_train.columns).sort_values(ascending=False)

plt.figure(figsize=(7, 5))
feat_imp.plot(kind="barh", color="#4C72B0")
plt.gca().invert_yaxis()
plt.title("Feature Importance (Random Forest)")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "12_feature_importance.png"), dpi=110)
plt.close()

print("\nFeature importance ranking:")
print(feat_imp.round(4))

# Save best model
joblib.dump(best_model, os.path.join(BASE_DIR, "best_model.joblib"))
print(f"\nSaved best model ({best_name}) to best_model.joblib")
