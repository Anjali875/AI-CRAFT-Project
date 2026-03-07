import pandas as pd
from preprocessing import preprocess
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

#Training Randomforest Model:
X_train, X_test, y_train, y_test, features, scaler = preprocess()

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nRandom Forest Accuracy: {accuracy:.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

feature_importance = pd.DataFrame({
    'feature': features,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)
print("\nTop 5 Important Features:")
print(feature_importance.head())


os.makedirs('models', exist_ok=True)
joblib.dump(rf_model, 'models/rf_model.pkl')


results = {
    'model': 'Random Forest',
    'accuracy': accuracy,
    'feature_importance': feature_importance.to_dict()
}
joblib.dump(results, 'models/rf_results.pkl')