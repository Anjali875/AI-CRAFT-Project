import pandas as pd
from preprocessing import preprocess
import joblib
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

#Training XGBOOST model:
X_train, X_test, y_train, y_test, features, scaler = preprocess()

xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)
xgb_model.fit(X_train, y_train)

y_pred = xgb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nXGBoost Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

feature_importance = pd.DataFrame({
    'feature': features,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)
print("\nTop 5 Important Features:")
print(feature_importance.head())

os.makedirs('models', exist_ok=True)
joblib.dump(xgb_model, 'models/xgb_model.pkl')
print("\nModel saved as 'models/xgb_model.pkl'")

results = {
    'model': 'XGBoost',
    'accuracy': accuracy,
    'feature_importance': feature_importance.to_dict()
}
joblib.dump(results, 'models/xgb_results.pkl')