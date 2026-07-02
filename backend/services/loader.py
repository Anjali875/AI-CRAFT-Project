import pickle
import joblib
import xgboost as xgb
from pathlib import Path

model_store = {}

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"

def load_models():
    print("Loading models...")

    pcos_model = xgb.XGBClassifier()
    pcos_model.load_model(MODELS_DIR / "xgb_model_new.json")
    model_store["pcos_model"] = pcos_model
    print("PCOS XGBoost model loaded")

    model_store["pcos_scaler"] = joblib.load(MODELS_DIR / "scaler_new.pkl")
    print("PCOS scaler loaded")

    with open(MODELS_DIR / "feature_names.pkl", "rb") as f:
        model_store["pcos_features"] = pickle.load(f)
    print("PCOS features loaded:", model_store["pcos_features"])

    model_store["endo_model"] = joblib.load(
    MODELS_DIR / "endo_logistic_model.pkl"
)
    print("Endometriosis Logistic Regression model loaded")

    model_store["endo_scaler"] = joblib.load(
    MODELS_DIR / "endo_scaler.pkl"
)
    print("Endometriosis scaler loaded")

    with open(MODELS_DIR / "endo_feature_names.pkl", "rb") as f:
        model_store["endo_features"] = pickle.load(f)
    print("Endo features loaded:", model_store["endo_features"])

    print("All models loaded successfully")