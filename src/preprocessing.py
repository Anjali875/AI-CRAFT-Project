import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

#Pre_processing data (encoding,handling missing values, scaling, train_test_splitting):
Target = 'PCOS (Y/N)'

SELECTED_FEATURES = [
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
    'Reg.Exercise(Y/N)'
]

def preprocess(csv_file="Data/PCOS_data_final_merged.csv", test_size=0.2, random_state=42, save_scaler=True):
    df = pd.read_csv(csv_file)
    print(f"   Shape: {df.shape}")
    imp_cols = SELECTED_FEATURES + [Target]
    df = df[imp_cols].copy()
    print(f"   Final columns: {list(df.columns)}")
    
    le = LabelEncoder()
    df[Target] = le.fit_transform(df[Target])
    binary_cols = [col for col in SELECTED_FEATURES if '(Y/N)' in col]
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0, 1: 1, 0: 0})
    
    numeric_cols = [' Age (yrs)', 'Weight (Kg)', 'Height(Cm) ', 'BMI', 'Cycle length(days)']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    if 'Cycle length(days)' in df.columns:
        df['Cycle length(days)'].fillna(28, inplace=True)
    
    for col in numeric_cols:
        if col in df.columns and df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)
    
    for col in binary_cols:
        if col in df.columns and df[col].isnull().any():
            df[col].fillna(0, inplace=True)
    
    print("\nSplitting features and target...")
    X = df.drop(columns=[Target])
    y = df[Target]
    
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
    
    numeric_to_scale = [col for col in numeric_cols if col in X.columns]
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[numeric_to_scale] = scaler.fit_transform(X_train[numeric_to_scale])
    X_test_scaled[numeric_to_scale] = scaler.transform(X_test[numeric_to_scale])
    
    #Saving Scaler and .pkl files:
    if save_scaler:
        joblib.dump(scaler, 'scaler.pkl')
        joblib.dump(list(X.columns), 'feature_names.pkl')
        print("\naved: scaler.pkl and feature_names.pkl")
    
    print("\n")
    print("PREPROCESSING COMPLETE!")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, list(X.columns), scaler