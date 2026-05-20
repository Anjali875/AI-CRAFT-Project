import pandas as pd
from preprocessing2 import preprocess_endometriosis
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import os

X_train, X_test, y_train, y_test, features, scaler = preprocess_endometriosis()

print("\nTraining Random Forest Model...\n")

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

base_model = RandomForestClassifier(
    random_state=42,
    n_jobs=-1
)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1,
    verbose=2
)

# TRAIN RF MODEL
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

print("\nBest Parameters:")
print(grid_search.best_params_)

print(f"\nBest CV Accuracy: {grid_search.best_score_:.3f}")

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"\nRandom Forest Test Accuracy: {accuracy:.3f}")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

feature_importance = pd.DataFrame({
    "Feature": features,
    "Importance": best_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nTop Important Features:\n")
print(feature_importance.head(10))

os.makedirs("models", exist_ok=True)

joblib.dump(best_model, "models/endo_rf_model.pkl")

results = {
    "model": "Random Forest",
    "accuracy": accuracy,
    "best_params": grid_search.best_params_,
    "best_cv_accuracy": grid_search.best_score_,
    "feature_importance": feature_importance.to_dict()
}

joblib.dump(results, "models/endo_rf_results.pkl")

print("\nSaved:")
print("   endo_rf_model.pkl")
print("   endo_rf_results.pkl")

print("\nRANDOM FOREST TRAINING COMPLETE!")