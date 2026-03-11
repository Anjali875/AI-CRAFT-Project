from preprocessing import preprocess
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

X_train, X_test, y_train, y_test, features, scaler = preprocess()

# Training Logistic Regression:
logistic_model = LogisticRegression(random_state=42, max_iter=300)
logistic_model.fit(X_train, y_train)

y_pred = logistic_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nLogistic Regression Accuracy: {accuracy:.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
  

os.makedirs('models', exist_ok=True)
joblib.dump(logistic_model, 'models/logistic_model.pkl')


results = {
    'model': 'Logistic Regression',
    'accuracy': accuracy,
    'report': classification_report(y_test, y_pred, output_dict=True)
}
joblib.dump(results, 'models/logistic_results.pkl')