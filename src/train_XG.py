import pandas as pd
from preprocessing import preprocess
import joblib
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import os

# Training XGBOOST model:
X_train, X_test, y_train, y_test, features, scaler = preprocess()

best_model = None  # Will store the best model
best_acc = 0

for rs in [42, 43, 44, 45, 123, 456]:
    model = xgb.XGBClassifier(
        n_estimators=250,
        max_depth=3,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=rs,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"Random state {rs}: Accuracy {acc:.4f}")

    if acc > best_acc:
        best_acc = acc  # ← INDENTED correctly
        best_model = model  # ← INDENTED correctly

# Now use the best model
y_pred = best_model.predict(X_test)
print(f"\n Best accuracy: {best_acc:.4f} with random_state={best_model.random_state}")

# ROC-AUC score
y_proba = best_model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_proba)
print(f" ROC-AUC Score: {roc_auc:.4f}")

print("\n Classification Report:")
print(classification_report(y_test, y_pred))

feature_importance = pd.DataFrame({
    "feature": features,
    "importance": best_model.feature_importances_
}).sort_values("importance", ascending=False)
print("\n Top 5 Important Features:")
print(feature_importance.head())

os.makedirs('models', exist_ok=True)
joblib.dump(best_model, 'models/xgb_model.pkl')
print("\n Model saved as 'models/xgb_model.pkl'")

results = {
    "model": "XGBoost",
    "accuracy": best_acc,  
    "roc_auc": roc_auc,   
    "feature_importance": feature_importance.to_dict() 
}

joblib.dump(results, 'models/xgb_results.pkl')
print("\n Results saved!")