from preprocessing2 import preprocess_endometriosis
import joblib
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import os

X_train, X_test, y_train, y_test, features, scaler = preprocess_endometriosis()

param_grid = {
	"n_estimators": [200, 400],
	"max_depth": [3, 5],
	"learning_rate": [0.03, 0.1],
	"subsample": [0.8, 1.0],
	"colsample_bytree": [0.8, 1.0],
}

base_model = xgb.XGBClassifier(
	random_state=42,
	use_label_encoder=False,
	eval_metric="logloss",
)
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
print(f"\nXGBoost Accuracy: {accuracy:.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

y_proba = best_model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_proba)
print(f"ROC-AUC Score: {roc_auc:.3f}")

os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/endo_xgb_model.pkl")

results = {
	"model": "XGBoost",
	"accuracy": accuracy,
	"roc_auc": roc_auc,
	"best_params": grid_search.best_params_,
	"best_cv_accuracy": grid_search.best_score_,
}
joblib.dump(results, "models/endo_xgb_results.pkl")
