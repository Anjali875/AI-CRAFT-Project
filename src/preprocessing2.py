from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

TARGET_COL = "Endometriosis_Stage"

SELECTED_FEATURES = [
	"Age",
	"BMI",
	"Cycle_Length",
	"Age_of_Menarche",
	"Dysmenorrhea_Score",
	"Urinary_Symptoms_Score",
	"Family_History",
	"Infertility_Status",
	"Mental_Health_Score",
	"Endometriosis_Stage",
]

MODEL_DIR = Path("models")


def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> None:
	for col in cols:
		if col in df.columns:
			df[col] = pd.to_numeric(df[col], errors="coerce")


def _coerce_binary(df: pd.DataFrame, cols: list[str]) -> None:
	mapping = {"Yes": 1, "No": 0, 1: 1, 0: 0, True: 1, False: 0}
	for col in cols:
		if col in df.columns:
			df[col] = df[col].map(mapping)


def preprocess_endometriosis(
	csv_file: str = "Data/endometriosis_data.csv",
	test_size: float = 0.2,
	random_state: int = 42,
	save_scaler: bool = True,
):
	df = pd.read_csv(csv_file)
	print(f"   Shape: {df.shape}")

	df = df[SELECTED_FEATURES].copy()
	print(f"   Final columns: {list(df.columns)}")

	binary_cols = ["Family_History", "Infertility_Status"]
	numeric_cols = [
		"Age",
		"BMI",
		"Cycle_Length",
		"Age_of_Menarche",
		"Dysmenorrhea_Score",
		"Urinary_Symptoms_Score",
		"Mental_Health_Score",
	]

	_coerce_binary(df, binary_cols)
	_coerce_numeric(df, numeric_cols)

	for col in numeric_cols:
		if col in df.columns and df[col].isnull().any():
			df[col].fillna(df[col].median(), inplace=True)

	for col in binary_cols:
		if col in df.columns and df[col].isnull().any():
			df[col].fillna(0, inplace=True)

	if TARGET_COL in df.columns:
		if not pd.api.types.is_numeric_dtype(df[TARGET_COL]):
			df[TARGET_COL] = LabelEncoder().fit_transform(df[TARGET_COL])

	print("\nSplitting features and target...")
	X = df.drop(columns=[TARGET_COL])
	y = df[TARGET_COL]

	print(f"   Features: {X.shape}")
	print(f"   Target: {y.value_counts().to_dict()}")

	print("\nTrain-test split...")
	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=test_size, random_state=random_state, stratify=y
	)
	print(f"   Train: {X_train.shape[0]} samples")
	print(f"   Test: {X_test.shape[0]} samples")

	print("\nScaling features:")
	scaler = StandardScaler()
	X_train_scaled = X_train.copy()
	X_test_scaled = X_test.copy()
	X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
	X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

	if save_scaler:
		MODEL_DIR.mkdir(exist_ok=True)
		joblib.dump(scaler, MODEL_DIR / "endo_scaler.pkl")
		joblib.dump(list(X.columns), MODEL_DIR / "endo_feature_names.pkl")
		print("\nSaved: endo_scaler.pkl and endo_feature_names.pkl")

	print("\nPREPROCESSING COMPLETE!")
	return X_train_scaled, X_test_scaled, y_train, y_test, list(X.columns), scaler


if __name__ == "__main__":
	preprocess_endometriosis()
