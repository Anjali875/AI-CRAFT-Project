import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
import os

print("MODEL COMPARISON")

from preprocessing import preprocess
X_train, X_test, y_train, y_test, features, scaler = preprocess()

models = {
    'Logistic Regression': joblib.load('models/logistic_model.pkl'),
    'Random Forest': joblib.load('models/rf_model.pkl'),
    'XGBoost': joblib.load('models/xgb_model.pkl')
}

results = []
for name, model in models.items():
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_proba)
    })

comparison_df = pd.DataFrame(results).round(4)
comparison_df.to_csv('models/model_comparison.csv', index=False)
print("\nComparison saved")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
for i, metric in enumerate(metrics):
    ax = axes[i//2, i%2]
    bars = ax.bar(comparison_df['Model'], comparison_df[metric])
    ax.set_title(metric)
    ax.set_ylim(0, 1)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.3f}', 
                ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('models/performance_metrics.png', dpi=150)
plt.show()


plt.figure(figsize=(8, 6))
for name, model in models.items():
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f'{name} (AUC={roc_auc_score(y_test, y_proba):.3f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('models/roc_curves.png', dpi=150)
plt.show()


fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for idx, name in enumerate(['Random Forest', 'XGBoost']):
    model = models[name]
    importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=True).tail(10)
    
    axes[idx].barh(importance['feature'], importance['importance'])
    axes[idx].set_title(f'{name} - Top 10 Features')

plt.tight_layout()
plt.savefig('models/feature_importance.png', dpi=150)
plt.show()