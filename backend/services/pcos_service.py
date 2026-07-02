import numpy as np
import pandas as pd
from services.loader import model_store

PCOS_NUMERIC_COLS = [
    ' Age (yrs)',
    'Weight (Kg)',
    'Height(Cm) ',
    'BMI',
    'Cycle length(days)',
]

PCOS_FEATURE_ORDER = [
    ' Age (yrs)',
    'Weight (Kg)',
    'Height(Cm) ',
    'BMI',
    'Cycle length(days)',
    'Weight gain(Y/N)',
    'hair growth(Y/N)',
    'Skin darkening (Y/N)',
    'Hair loss(Y/N)',
    'Pimples(Y/N)',
    'Fast food (Y/N)',
    'Reg.Exercise(Y/N)',
]

PCOS_CONTRIBUTING_FACTORS = {
    'Cycle length(days)':   'Period duration',
    'BMI':                  'BMI',
    'hair growth(Y/N)':     'Excess hair growth',
    'Weight gain(Y/N)':     'Unexplained weight gain',
    'Skin darkening (Y/N)': 'Skin darkening',
    'Pimples(Y/N)':         'Frequent acne',
    'Hair loss(Y/N)':       'Hair loss',
    'Fast food (Y/N)':      'Fast food consumption',
    'Weight (Kg)':          'Weight',
    ' Age (yrs)':           'Age',
    'Reg.Exercise(Y/N)':    'Exercise habits',
    'Height(Cm) ':          'Height',
}

def get_risk_level(prob: float) -> str:
    if prob < 0.35:
        return "Low"
    elif prob < 0.65:
        return "Moderate"
    else:
        return "High"

def get_contributing_factors(input_dict: dict) -> list:
    factors = []

    if input_dict.get('Cycle length(days)', 0) > 7:
        factors.append('Period duration')
    if input_dict.get('BMI', 0) > 25:
        factors.append('BMI')
    if input_dict.get('hair growth(Y/N)', 0) == 1:
        factors.append('Excess hair growth')
    if input_dict.get('Weight gain(Y/N)', 0) == 1:
        factors.append('Unexplained weight gain')
    if input_dict.get('Skin darkening (Y/N)', 0) == 1:
        factors.append('Skin darkening')
    if input_dict.get('Pimples(Y/N)', 0) == 1:
        factors.append('Frequent acne')
    if input_dict.get('Hair loss(Y/N)', 0) == 1:
        factors.append('Hair loss')
    if input_dict.get('Fast food (Y/N)', 0) == 1:
        factors.append('Fast food consumption')
    if input_dict.get('Reg.Exercise(Y/N)', 0) == 0:
        factors.append('Low physical activity')

    return factors[:4]

def predict_pcos(data):
    model = model_store["pcos_model"]
    scaler = model_store["pcos_scaler"]

    raw = {
        ' Age (yrs)':           data.age,
        'Weight (Kg)':          data.weight,
        'Height(Cm) ':          data.height,
        'BMI':                  data.bmi,
        'Cycle length(days)':   data.cycle_length,
        'Weight gain(Y/N)':     data.weight_gain,
        'hair growth(Y/N)':     data.hair_growth,
        'Skin darkening (Y/N)': data.skin_darkening,
        'Hair loss(Y/N)':       data.hair_loss,
        'Pimples(Y/N)':         data.pimples,
        'Fast food (Y/N)':      data.fast_food,
        'Reg.Exercise(Y/N)':    data.regular_exercise,
    }

    df = pd.DataFrame([raw], columns=PCOS_FEATURE_ORDER)

    df[PCOS_NUMERIC_COLS] = scaler.transform(df[PCOS_NUMERIC_COLS])

    prob = float(model.predict_proba(df)[0][1])
    level = get_risk_level(prob)
    factors = get_contributing_factors(raw)

    return {
        "condition": "pcos",
        "probability": round(prob, 4),
        "risk_level": level,
        "risk_percentage": round(prob * 100, 1),
        "contributing_factors": factors,
    }