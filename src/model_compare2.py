import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
	accuracy_score,
	precision_score,
	recall_score,
	f1_score,
	roc_auc_score,
	roc_curve,
)
import os

print("MODEL COMPARISON - ENDOMETRIOSIS")

from preprocessing2 import preprocess_endometriosis

X_train, X_test, y_train, y_test, features, scaler = preprocess_endometriosis()

models = {
	"Logistic Regression": joblib.load("models/endo_logistic_model.pkl"),
	"Random Forest": joblib.load("models/endo_rf_model.pkl"),
	"XGBoost": joblib.load("models/endo_xgb_model.pkl"),
}

results = []
class_count = len(np.unique(y_test))
is_binary = class_count == 2

for name, model in models.items():
	y_pred = model.predict(X_test)
	y_proba = model.predict_proba(X_test)

	precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
	recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
	f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

	if is_binary:
		roc_auc = roc_auc_score(y_test, y_proba[:, 1])
	else:
		roc_auc = roc_auc_score(
			y_test,
			y_proba,
			multi_class="ovr",
			average="weighted",
		)

	results.append(
		{
			"Model": name,
			"Accuracy": accuracy_score(y_test, y_pred),
			"Precision": precision,
			"Recall": recall,
			"F1-Score": f1,
			"ROC-AUC": roc_auc,
		}
	)

comparison_df = pd.DataFrame(results).round(4)
comparison_df.to_csv("models/model_comparison2.csv", index=False)
print("\nComparison saved")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
for i, metric in enumerate(metrics):
	ax = axes[i // 2, i % 2]
	bars = ax.bar(comparison_df["Model"], comparison_df[metric])
	ax.set_title(metric)
	ax.set_ylim(0, 1)
	for bar in bars:
		height = bar.get_height()
		ax.text(
			bar.get_x() + bar.get_width() / 2.0,
			height,
			f"{height:.3f}",
			ha="center",
			va="bottom",
			fontsize=8,
		)

plt.tight_layout()
plt.savefig("models/performance_metrics_endo.png", dpi=150)
plt.show()

if is_binary:
	plt.figure(figsize=(8, 6))
	for name, model in models.items():
		y_proba = model.predict_proba(X_test)[:, 1]
		fpr, tpr, _ = roc_curve(y_test, y_proba)
		plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, y_proba):.3f})")

	plt.plot([0, 1], [0, 1], "k--")
	plt.xlabel("False Positive Rate")
	plt.ylabel("True Positive Rate")
	plt.title("ROC Curves")
	plt.legend()
	plt.grid(True, alpha=0.3)
	plt.savefig("models/roc_curves_endo.png", dpi=150)
	plt.show()
else:
	print("\nROC curves skipped (multiclass target).")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for idx, name in enumerate(["Random Forest", "XGBoost"]):
	model = models[name]
	importance = pd.DataFrame(
		{"feature": features, "importance": model.feature_importances_}
	).sort_values("importance", ascending=True).tail(10)

	axes[idx].barh(importance["feature"], importance["importance"])
	axes[idx].set_title(f"{name} - Top 10 Features")

plt.tight_layout()
plt.savefig("models/feature_importance_endo.png", dpi=150)
plt.show()
