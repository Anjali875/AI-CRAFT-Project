from preprocessing2 import preprocess_endometriosis
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import os

X_train, X_test, y_train, y_test, features, scaler = preprocess_endometriosis()

# Tune Logistic Regression model with grid search + cross-validation
param_grid = {
	"C": [0.01, 0.1, 1.0, 10.0],
	"penalty": ["l1", "l2"],
	"solver": ["liblinear", "saga"],
	"class_weight": [None, "balanced"],
}

base_model = LogisticRegression(random_state=42, max_iter=1000)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(
	base_model,
	param_grid,
	cv=cv,
	scoring="accuracy",
	n_jobs=-1,
)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print("\nBest params:")
print(grid_search.best_params_)
print(f"Best CV accuracy: {grid_search.best_score_:.3f}")

y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nLogistic Regression Accuracy: {accuracy:.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/endo_logistic_model.pkl")

results = {
	"model": "Logistic Regression",
	"accuracy": accuracy,
	"best_params": grid_search.best_params_,
	"best_cv_accuracy": grid_search.best_score_,
	"report": classification_report(y_test, y_pred, output_dict=True),
}
joblib.dump(results, "models/endo_logistic_results.pkl")