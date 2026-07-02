import numpy as np
import pandas as pd
from services.loader import model_store

ENDO_NUMERIC_COLS = [
    'Age',
    'BMI',
    'Cycle_Length',
    'Age_of_Menarche',
    'Dysmenorrhea_Score',
    'Urinary_Symptoms_Score',
    'Mental_Health_Score',
]

ENDO_FEATURE_ORDER = [
    'Age',
    'BMI',
    'Cycle_Length',
    'Age_of_Menarche',
    'Dysmenorrhea_Score',
    'Urinary_Symptoms_Score',
    'Family_History',
    'Infertility_Status',
    'Mental_Health_Score',
]

def get_risk_level(prob: float) -> str:
    if prob < 0.35:
        return "Low"
    elif prob < 0.65:
        return "Moderate"
    else:
        return "High"

def get_contributing_factors(input_dict: dict) -> list:
    factors = []

    if input_dict.get('Dysmenorrhea_Score', 0) >= 7:
        factors.append('Severe menstrual pain')
    elif input_dict.get('Dysmenorrhea_Score', 0) >= 4:
        factors.append('Moderate menstrual pain')

    if input_dict.get('Mental_Health_Score', 0) >= 7:
        factors.append('High impact on mental wellbeing')

    if input_dict.get('Urinary_Symptoms_Score', 0) >= 5:
        factors.append('Urinary symptoms present')

    if input_dict.get('Family_History', 0) == 1:
        factors.append('Family history of endometriosis')

    if input_dict.get('Infertility_Status', 0) == 1:
        factors.append('Fertility challenges reported')

    if input_dict.get('BMI', 0) < 18.5:
        factors.append('Low BMI')
    elif input_dict.get('BMI', 0) > 25:
        factors.append('Elevated BMI')

    if input_dict.get('Cycle_Length', 0) > 35:
        factors.append('Longer menstrual cycle')

    return factors[:4]

def predict_endo(data):
    model = model_store["endo_model"]
    scaler = model_store["endo_scaler"]

    bmi = round(data.weight / ((data.height / 100) ** 2), 2)

    raw = {
        'Age':                    data.age,
        'BMI':                    bmi,
        'Cycle_Length':           data.cycle_length,
        'Age_of_Menarche':        data.age_of_menarche,
        'Dysmenorrhea_Score':     data.dysmenorrhea_score,
        'Urinary_Symptoms_Score': data.urinary_symptoms_score,
        'Family_History':         data.family_history,
        'Infertility_Status':     data.infertility_status,
        'Mental_Health_Score':    data.mental_health_score,
    }

    df = pd.DataFrame([raw], columns=ENDO_FEATURE_ORDER)

    df[ENDO_NUMERIC_COLS] = scaler.transform(df[ENDO_NUMERIC_COLS])

    prob = float(model.predict_proba(df)[0][1])
    level = get_risk_level(prob)
    factors = get_contributing_factors(raw)

    return {
        "condition": "endo",
        "probability": round(prob, 4),
        "risk_level": level,
        "risk_percentage": round(prob * 100, 1),
        "contributing_factors": factors,
    }