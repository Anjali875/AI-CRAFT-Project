import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import os

st.set_page_config(
    page_title="PCOS Risk Analyzer",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


ILLUS_HOME = """
<svg viewBox="0 0 200 180" xmlns="http://www.w3.org/2000/svg" width="180">
  <!-- body -->
  <ellipse cx="100" cy="130" rx="38" ry="48" fill="#ddd6fe"/>
  <!-- head -->
  <circle cx="100" cy="68" r="28" fill="#fde68a"/>
  <!-- hair -->
  <ellipse cx="100" cy="46" rx="28" ry="14" fill="#7c3aed"/>
  <ellipse cx="76"  cy="60" rx="8"  ry="16" fill="#7c3aed"/>
  <ellipse cx="124" cy="60" rx="8"  ry="16" fill="#7c3aed"/>
  <!-- eyes -->
  <circle cx="91" cy="68" r="3" fill="#3b0764"/>
  <circle cx="109" cy="68" r="3" fill="#3b0764"/>
  <!-- smile -->
  <path d="M92 78 Q100 86 108 78" stroke="#3b0764" stroke-width="2" fill="none" stroke-linecap="round"/>
  <!-- stethoscope -->
  <path d="M88 100 Q80 120 85 135 Q90 148 100 148 Q110 148 115 135 Q120 120 112 100"
        stroke="#7c3aed" stroke-width="3" fill="none" stroke-linecap="round"/>
  <circle cx="100" cy="150" r="6" fill="#a855f7" stroke="#7c3aed" stroke-width="2"/>
  <!-- arms -->
  <ellipse cx="68"  cy="118" rx="10" ry="24" fill="#ddd6fe" transform="rotate(-15 68 118)"/>
  <ellipse cx="132" cy="118" rx="10" ry="24" fill="#ddd6fe" transform="rotate(15 132 118)"/>
  <!-- clipboard in right hand -->
  <rect x="118" y="118" width="22" height="28" rx="3" fill="white" stroke="#c4b5fd" stroke-width="1.5"/>
  <line x1="122" y1="126" x2="136" y2="126" stroke="#a78bfa" stroke-width="1.5"/>
  <line x1="122" y1="131" x2="136" y2="131" stroke="#a78bfa" stroke-width="1.5"/>
  <line x1="122" y1="136" x2="130" y2="136" stroke="#a78bfa" stroke-width="1.5"/>
</svg>"""

ILLUS_CHART = """
<svg viewBox="0 0 200 160" xmlns="http://www.w3.org/2000/svg" width="180">
  <!-- background card -->
  <rect x="10" y="10" width="180" height="140" rx="16" fill="white" stroke="#c4b5fd" stroke-width="2"/>
  <!-- bar chart bars -->
  <rect x="30"  y="90"  width="22" height="45" rx="4" fill="#c4b5fd"/>
  <rect x="62"  y="65"  width="22" height="70" rx="4" fill="#a78bfa"/>
  <rect x="94"  y="45"  width="22" height="90" rx="4" fill="#8b5cf6"/>
  <rect x="126" y="55"  width="22" height="80" rx="4" fill="#7c3aed"/>
  <rect x="158" y="75"  width="22" height="60" rx="4" fill="#6d28d9"/>
  <!-- x axis -->
  <line x1="20" y1="135" x2="185" y2="135" stroke="#e9d5ff" stroke-width="2"/>
  <!-- trend line -->
  <polyline points="41,112 73,88 105,67 137,75 169,95"
            stroke="#f59e0b" stroke-width="2.5" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
  <!-- dots on trend -->
  <circle cx="41"  cy="112" r="3.5" fill="#f59e0b"/>
  <circle cx="73"  cy="88"  r="3.5" fill="#f59e0b"/>
  <circle cx="105" cy="67"  r="3.5" fill="#f59e0b"/>
  <circle cx="137" cy="75"  r="3.5" fill="#f59e0b"/>
  <circle cx="169" cy="95"  r="3.5" fill="#f59e0b"/>
  <!-- title text -->
  <text x="100" y="30" text-anchor="middle" font-size="11" fill="#7c3aed" font-weight="bold">Model Performance</text>
</svg>"""

ILLUS_CHAT = """
<svg viewBox="0 0 200 160" xmlns="http://www.w3.org/2000/svg" width="170">
  <!-- bot body -->
  <rect x="60" y="55" width="80" height="70" rx="18" fill="#ede9fe" stroke="#a78bfa" stroke-width="2"/>
  <!-- bot head -->
  <rect x="70" y="25" width="60" height="45" rx="14" fill="#ddd6fe" stroke="#a78bfa" stroke-width="2"/>
  <!-- antenna -->
  <line x1="100" y1="25" x2="100" y2="12" stroke="#8b5cf6" stroke-width="2.5" stroke-linecap="round"/>
  <circle cx="100" cy="9" r="5" fill="#a855f7"/>
  <!-- eyes -->
  <rect x="80" y="36" width="14" height="10" rx="5" fill="#7c3aed"/>
  <rect x="106" y="36" width="14" height="10" rx="5" fill="#7c3aed"/>
  <!-- eye shine -->
  <circle cx="85"  cy="39" r="2.5" fill="white"/>
  <circle cx="111" cy="39" r="2.5" fill="white"/>
  <!-- mouth panel -->
  <rect x="82" y="52" width="36" height="12" rx="6" fill="#c4b5fd"/>
  <line x1="88" y1="58" x2="92"  y2="58" stroke="white" stroke-width="1.5" stroke-linecap="round"/>
  <line x1="96" y1="58" x2="100" y2="58" stroke="white" stroke-width="1.5" stroke-linecap="round"/>
  <line x1="104" y1="58" x2="108" y2="58" stroke="white" stroke-width="1.5" stroke-linecap="round"/>
  <!-- arms -->
  <rect x="32" y="70" width="30" height="12" rx="6" fill="#ddd6fe" stroke="#a78bfa" stroke-width="1.5"/>
  <rect x="138" y="70" width="30" height="12" rx="6" fill="#ddd6fe" stroke="#a78bfa" stroke-width="1.5"/>
  <!-- chat bubble coming from bot -->
  <rect x="20" y="100" width="80" height="30" rx="10" fill="#7c3aed"/>
  <polygon points="55,130 65,130 58,142" fill="#7c3aed"/>
  <text x="60" y="119" text-anchor="middle" font-size="9" fill="white">How can I help</text>
  <text x="60" y="129" text-anchor="middle" font-size="9" fill="white">you today? 😊</text>
  <!-- legs -->
  <rect x="78"  y="122" width="16" height="24" rx="8" fill="#ddd6fe" stroke="#a78bfa" stroke-width="1.5"/>
  <rect x="106" y="122" width="16" height="24" rx="8" fill="#ddd6fe" stroke="#a78bfa" stroke-width="1.5"/>
</svg>"""

ILLUS_PREDICT = """
<svg viewBox="0 0 200 130" xmlns="http://www.w3.org/2000/svg" width="180">
  <!-- gauge arc background -->
  <path d="M20 110 A80 80 0 0 1 180 110" stroke="#e9d5ff" stroke-width="18" fill="none" stroke-linecap="round"/>
  <!-- gauge low zone -->
  <path d="M20 110 A80 80 0 0 1 73 38" stroke="#d1fae5" stroke-width="18" fill="none" stroke-linecap="round"/>
  <!-- gauge mid zone -->
  <path d="M73 38 A80 80 0 0 1 127 38" stroke="#fef3c7" stroke-width="18" fill="none" stroke-linecap="round"/>
  <!-- gauge high zone -->
  <path d="M127 38 A80 80 0 0 1 180 110" stroke="#fee2e2" stroke-width="18" fill="none" stroke-linecap="round"/>
  <!-- needle -->
  <line x1="100" y1="110" x2="52" y2="52" stroke="#7c3aed" stroke-width="3.5" stroke-linecap="round"/>
  <circle cx="100" cy="110" r="8" fill="#7c3aed"/>
  <circle cx="100" cy="110" r="4" fill="white"/>
  <!-- labels -->
  <text x="18"  y="128" font-size="10" fill="#065f46" font-weight="bold">LOW</text>
  <text x="82"  y="25"  font-size="10" fill="#92400e" font-weight="bold">MED</text>
  <text x="158" y="128" font-size="10" fill="#991b1b" font-weight="bold">HIGH</text>
  <!-- title -->
  <text x="100" y="100" text-anchor="middle" font-size="11" fill="#7c3aed" font-weight="bold">PCOS Risk</text>
</svg>"""

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp { background: #f5f3ff; }
    .main .block-container { padding-top: 1.5rem; }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ede9fe 0%, #ddd6fe 60%, #ede9fe 100%);
    }
    [data-testid="stSidebar"] * { color: #3b0764 !important; }
    [data-testid="stSidebar"] .stRadio label {
        font-size: 15px !important;
        padding: 6px 0 !important;
        color: #4c1d95 !important;
    }

    .card {
        background: #ffffff;
        border: 1px solid #c4b5fd;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 16px;
        box-shadow: 0 4px 16px rgba(139,92,246,0.10);
    }
    .card-title {
        color: #7c3aed;
        font-size: 18px;
        font-weight: 600;
        margin-bottom: 12px;
    }

    .badge-low  { background:#d1fae5; color:#065f46; border:1.5px solid #6ee7b7; padding:6px 18px; border-radius:20px; font-weight:600; }
    .badge-mod  { background:#fef3c7; color:#92400e; border:1.5px solid #fbbf24; padding:6px 18px; border-radius:20px; font-weight:600; }
    .badge-high { background:#fee2e2; color:#991b1b; border:1.5px solid #f87171; padding:6px 18px; border-radius:20px; font-weight:600; }

    .feature-pill {
        display:inline-block;
        background:#ede9fe;
        color:#5b21b6;
        border:1px solid #a78bfa;
        border-radius:12px;
        padding:4px 14px;
        margin:4px;
        font-size:13px;
    }

    label { color: #4c1d95 !important; font-size:14px !important; }

    .stButton > button {
        background: linear-gradient(135deg, #7c3aed, #a855f7) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 10px 30px !important;
        font-weight: 600 !important;
        font-size: 15px !important;
        width: 100%;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(124,58,237,0.35) !important;
    }
    /* Quick reply pill style */
    .quick-btn .stButton > button {
        background: #ede9fe !important;
        color: #5b21b6 !important;
        border: 1.5px solid #a78bfa !important;
        border-radius: 20px !important;
        padding: 5px 10px !important;
        font-size: 12px !important;
        font-weight: 500 !important;
    }
    .quick-btn .stButton > button:hover {
        background: #ddd6fe !important;
        box-shadow: 0 3px 8px rgba(124,58,237,0.2) !important;
    }

    h1,h2,h3 { color: #3b0764 !important; }
    p, li     { color: #4b5563 !important; }

    .chat-user {
        background:#ede9fe;
        border-radius:16px 16px 4px 16px;
        padding:12px 16px; margin:8px 0;
        color:#3b0764;
        text-align:right; margin-left:20%;
        border: 1px solid #c4b5fd;
    }
    .chat-bot {
        background:#f0fdf4;
        border-radius:16px 16px 16px 4px;
        padding:12px 16px; margin:8px 0;
        color:#065f46;
        margin-right:20%;
        border: 1px solid #86efac;
    }
    .chat-bot-warn {
        background:#fffbeb;
        border-radius:16px 16px 16px 4px;
        padding:12px 16px; margin:8px 0;
        color:#92400e;
        margin-right:20%;
        border: 1px solid #fcd34d;
    }
    .chat-bot-alert {
        background:#fff1f2;
        border-radius:16px 16px 16px 4px;
        padding:12px 16px; margin:8px 0;
        color:#991b1b;
        margin-right:20%;
        border: 1px solid #fca5a5;
    }

    .metric-box {
        background:#ffffff;
        border:1.5px solid #a78bfa;
        border-radius:12px; padding:18px; text-align:center;
        box-shadow: 0 2px 8px rgba(139,92,246,0.10);
    }
    .metric-val { color:#7c3aed; font-size:28px; font-weight:700; }
    .metric-lbl { color:#6b7280; font-size:13px; margin-top:4px; }

    /* illustration container */
    .illus-center { display:flex; justify-content:center; align-items:center; padding:10px 0; }

    hr { border-color:#e9d5ff !important; }

    ::-webkit-scrollbar { width:6px; }
    ::-webkit-scrollbar-track { background:#f5f3ff; }
    ::-webkit-scrollbar-thumb { background:#a78bfa; border-radius:3px; }

    .stDataFrame { border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

BASE = os.path.dirname(__file__)

@st.cache_resource
def load_assets():
    assets = {}
    model_dir = os.path.join(BASE, "models")

    for name in ["best_model", "xgb_model", "rf_model", "logistic_model"]:
        path = os.path.join(model_dir, f"{name}.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                assets["model"] = pickle.load(f)
            assets["model_name"] = name.replace("_model", "").upper()
            break

    scaler_path = os.path.join(model_dir, "scaler.pkl")
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            assets["scaler"] = pickle.load(f)

    feat_path = os.path.join(model_dir, "feature_names.pkl")
    if os.path.exists(feat_path):
        with open(feat_path, "rb") as f:
            assets["feature_names"] = pickle.load(f)

    for res in ["xgb_results", "rf_results", "logistic_results"]:
        path = os.path.join(model_dir, f"{res}.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                assets[res] = pickle.load(f)

    cmp_path = os.path.join(model_dir, "model_comparison.csv")
    if os.path.exists(cmp_path):
        assets["comparison_df"] = pd.read_csv(cmp_path)

    return assets

assets = load_assets()

NUMERIC_FEATURES = [
    ("Age (years)",         "age_years"),
    ("Weight (kg)",         "weight_kg"),
    ("Height (cm)",         "height_cm"),
    ("BMI",                 "bmi"),
    ("Cycle Length (days)", "cycle_length_days"),
]

BINARY_FEATURES = [
    ("Weight Gain",      "weight_gain"),
    ("Hair Growth",      "hair_growth"),
    ("Skin Darkening",   "skin_darkening"),
    ("Hair Loss",        "hair_loss"),
    ("Pimples",          "pimples"),
    ("Fast Food",        "fast_food"),
    ("Regular Exercise", "regular_exercise"),
]

FEATURE_ORDER = [
    "age_years", "weight_kg", "height_cm", "bmi", "cycle_length_days",
    "weight_gain", "hair_growth", "skin_darkening", "hair_loss",
    "pimples", "fast_food", "regular_exercise",
]

FEATURE_TIPS = {
    "weight_gain":       "Unexplained weight gain can indicate hormonal imbalance.",
    "hair_growth":       "Excess facial/body hair (hirsutism) is a common PCOS sign.",
    "skin_darkening":    "Acanthosis nigricans often linked to insulin resistance.",
    "hair_loss":         "Male-pattern hair thinning is a hormonal indicator.",
    "pimples":           "Hormonal acne, especially on jaw/chin, is associated with PCOS.",
    "fast_food":         "High-GI diet can worsen insulin resistance.",
    "regular_exercise":  "Exercise improves insulin sensitivity and hormonal balance.",
    "bmi":               "Higher BMI correlates with increased PCOS risk.",
    "cycle_length_days": "Irregular or long cycles are a primary PCOS diagnostic criterion.",
}

XGB_METRICS = {
    "Accuracy":  0.8684,
    "Precision": 0.8621,
    "Recall":    0.6944,
    "F1-Score":  0.7692,
    "ROC-AUC":   0.9152,
}

def predict_risk(input_dict):
    model  = assets.get("model")
    scaler = assets.get("scaler")
    if model is None:
        return None, None, None

    feat_vec = np.array([[input_dict[k] for k in FEATURE_ORDER]], dtype=float)

    if scaler is not None:
        try:
            feat_vec = scaler.transform(feat_vec)
        except Exception:
            pass

    prob = float(model.predict_proba(feat_vec)[0][1]) if hasattr(model, "predict_proba") else float(model.predict(feat_vec)[0])

    if prob < 0.35:
        level = "Low"
    elif prob < 0.65:
        level = "Moderate"
    else:
        level = "High"

   
    top_features = ["Cycle Length Days", "Bmi", "Hair Growth"]
    return prob, level, top_features

def make_gauge(prob, level):
    color = {"Low": "#10b981", "Moderate": "#f59e0b", "High": "#ef4444"}[level]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        number={"suffix": "%", "font": {"color": color, "size": 42}},
        title={"text": f"<b>PCOS Risk Score</b><br><span style='color:{color};font-size:18px'>{level} Risk</span>",
               "font": {"color": "#3b0764", "size": 16}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#9ca3af", "tickfont": {"color": "#6b7280"}},
            "bar":  {"color": color, "thickness": 0.28},
            "bgcolor": "#f9fafb",
            "bordercolor": "#e5e7eb",
            "steps": [
                {"range": [0,  35],  "color": "#d1fae5"},
                {"range": [35, 65],  "color": "#fef3c7"},
                {"range": [65, 100], "color": "#fee2e2"},
            ],
            "threshold": {"line": {"color": color, "width": 4}, "thickness": 0.8, "value": prob * 100},
        }
    ))
    fig.update_layout(
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        font_color="#3b0764", height=300,
        margin=dict(t=60, b=0, l=20, r=20)
    )
    return fig

with st.sidebar:
    st.markdown("## 🩺 PCOS Analyzer")
    st.markdown("---")
    page = st.radio("Navigate", [
        "🏠  Home",
        "🔮  Risk Prediction",
        "🤖  AI Chatbot",
        "📊  Model Insights",
        "ℹ️   About",
    ])
    st.markdown("---")
    st.markdown("**Model Active:**")
    st.markdown("<span style='color:#7c3aed;font-weight:700'>XGBoost</span>", unsafe_allow_html=True)
    st.markdown("**Features Used:** 12")
    st.markdown("---")
    st.markdown("<small style='color:#6b7280'>⚕️ For educational use only.<br>Always consult a doctor.</small>",
                unsafe_allow_html=True)

page = page.strip()

if "Home" in page:

    hero_col, title_col = st.columns([1, 2])
    with hero_col:
        st.markdown(f'<div class="illus-center">{ILLUS_HOME}</div>', unsafe_allow_html=True)
    with title_col:
        st.markdown("<h1 style='color:#3b0764;margin-top:30px'>🩺 PCOS Risk Analyzer</h1>", unsafe_allow_html=True)
        st.markdown("<p style='color:#7c3aed;font-size:17px'>AI-Powered Polycystic Ovary Syndrome<br>Early Risk Detection System</p>",
                    unsafe_allow_html=True)
        st.markdown("""
        <div style="margin-top:12px">
            <span style="background:#d1fae5;color:#065f46;padding:5px 14px;border-radius:20px;font-size:13px;margin-right:8px;font-weight:600">✅ Low Risk</span>
            <span style="background:#fef3c7;color:#92400e;padding:5px 14px;border-radius:20px;font-size:13px;margin-right:8px;font-weight:600">⚠️ Moderate Risk</span>
            <span style="background:#fee2e2;color:#991b1b;padding:5px 14px;border-radius:20px;font-size:13px;font-weight:600">🚨 High Risk</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Feature cards with illustrations
    c1, c2, c3 = st.columns(3)
    for col, illus, icon, title, desc in [
        (c1, ILLUS_PREDICT, "🔮", "Risk Prediction",  "Enter your health details and get an instant risk score with a gauge visualization."),
        (c2, ILLUS_CHAT,    "🤖", "AI Chatbot",       "Chat with our intelligent assistant that adapts its tone based on how you're feeling."),
        (c3, ILLUS_CHART,   "📊", "Model Insights",   "Explore XGBoost performance metrics, ROC curves, and feature importance charts."),
    ]:
        with col:
            st.markdown(f"""
            <div class="card" style="text-align:center">
                <div class="illus-center">{illus}</div>
                <div class="card-title" style="margin-top:8px">{icon} {title}</div>
                <p style="font-size:14px">{desc}</p>
            </div>""", unsafe_allow_html=True)

    st.markdown("### 📌 Key Features Analyzed")
    pills = ["Age", "Weight", "Height", "BMI", "Cycle Length",
             "Weight Gain", "Hair Growth", "Skin Darkening",
             "Hair Loss", "Pimples", "Fast Food", "Regular Exercise"]
    st.markdown(" ".join([f'<span class="feature-pill">{p}</span>' for p in pills]), unsafe_allow_html=True)

    st.markdown("---")
    # Banner with image and title overlay
    banner_col, text_col = st.columns([1, 2])
    with banner_col:
     st.image("assets/banner.png", width=280)
    with text_col:
     st.markdown("""
    <div style="background:linear-gradient(135deg,#ede9fe,#ddd6fe);
                border:1.5px solid #a78bfa;border-radius:20px;
                padding:30px 35px;margin-top:10px;">
        <div style="color:#7c3aed;font-size:13px;font-weight:600;
                    letter-spacing:2px;text-transform:uppercase">
            🤖 Machine Learning Powered
        </div>
        <div style="color:#3b0764;font-size:28px;font-weight:800;
                    margin:10px 0;line-height:1.3">
            AI-Powered PCOS<br>Risk Detection System
        </div>
        <div style="color:#6b7280;font-size:14px;margin-bottom:18px">
            Early awareness • Instant results • Personalised insights
        </div>
        <span style="background:#7c3aed;color:white;padding:8px 20px;
                     border-radius:20px;font-size:13px;font-weight:600">
            ✨ XGBoost Model • 86.8% Accuracy
        </span>
    </div>
    """, unsafe_allow_html=True)

    # Stats row
    s1, s2, s3, s4 = st.columns(4)
    for col, val, lbl, clr in [
        (s1, "86.8%", "XGBoost Accuracy", "#7c3aed"),
        (s2, "91.5%", "ROC-AUC Score",    "#10b981"),
        (s3, "86.2%", "Precision",         "#f59e0b"),
        (s4, "12",    "Features Used",     "#3b82f6"),
    ]:
    
        with col:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-val" style="color:{clr}">{val}</div>
                <div class="metric-lbl">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("")
    st.markdown("""
    <div class="card">
        <div class="card-title">⚠️ Medical Disclaimer</div>
        <p>This tool is designed for <b>educational and awareness purposes only</b>. It is not a substitute
        for professional medical diagnosis or treatment. If you have concerns about PCOS, please consult
        a qualified gynaecologist or endocrinologist.</p>
    </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════
#  PAGE 2 – RISK PREDICTION
# ═══════════════════════════════════════════════
elif "Prediction" in page:
    # Page header with illustration
    hdr_col, form_intro = st.columns([1, 3])
    with hdr_col:
        st.markdown(f'<div class="illus-center" style="margin-top:10px">{ILLUS_PREDICT}</div>',
                    unsafe_allow_html=True)
    with form_intro:
        st.markdown("## 🔮 PCOS Risk Prediction")
        st.markdown("Fill in your details below. Yes/No answers are auto-converted for the model.")
        st.markdown("""
        <div style="background:#ede9fe;border-radius:10px;padding:10px 16px;margin-top:8px;border:1px solid #c4b5fd">
            <span style="color:#4c1d95;font-size:13px">
            🤖 <b>Model:</b> XGBoost &nbsp;|&nbsp;
            📊 <b>Accuracy:</b> 86.8% &nbsp;|&nbsp;
            🎯 <b>ROC-AUC:</b> 91.5%
            </span>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    with st.form("prediction_form"):
        st.markdown("### 📐 Numeric Health Metrics")
        st.caption("Enter positive values only.")
        col1, col2, col3 = st.columns(3)
        cols   = [col1, col2, col3]
        inputs = {}

        for i, (label, key) in enumerate(NUMERIC_FEATURES):
            with cols[i % 3]:
                val = st.number_input(
                    label,
                    min_value=0.0,
                    value=None,
                    placeholder="Enter value...",
                    step=0.1,
                    format="%.2f"
                )
                inputs[key] = val if val is not None else 0.0
                tip = FEATURE_TIPS.get(key)
                if tip:
                    st.caption(f"💡 {tip}")

        st.markdown("### ✅ Symptom Checklist  *(Yes = 1 | No = 0)*")
        b_cols = st.columns(4)
        for i, (label, key) in enumerate(BINARY_FEATURES):
            with b_cols[i % 4]:
                ans = st.selectbox(label, ["No", "Yes"], key=key, index=0)
                inputs[key] = 1 if ans == "Yes" else 0
                tip = FEATURE_TIPS.get(key)
                if tip:
                    st.caption(f"💡 {tip}")

        submitted = st.form_submit_button("🔍 Analyze Risk")

    if submitted:
        if assets.get("model") is None:
            st.error("⚠️ No trained model found in the `models/` folder. Please ensure Anjali's model files are present.")
        else:
            prob, level, top_feats = predict_risk(inputs)
            st.session_state["last_prediction"] = {
                "prob": prob, "level": level,
                "top_feats": top_feats, "inputs": inputs
            }

            st.markdown("---")
            st.markdown("## 📊 Your Results")

            gauge_col, result_col = st.columns([1, 1])

            with gauge_col:
                st.plotly_chart(make_gauge(prob, level), use_container_width=True)

            with result_col:
                badge_map = {"Low": "badge-low", "Moderate": "badge-mod", "High": "badge-high"}
                emoji_map = {"Low": "✅", "Moderate": "⚠️", "High": "🚨"}

                st.markdown(f"""
                <div class="card">
                    <div class="card-title">Risk Assessment</div>
                    <div style="margin:12px 0">
                        <span class="{badge_map[level]}">{emoji_map[level]} {level} Risk</span>
                    </div>
                    <p style="font-size:14px;margin-top:10px">
                        Your predicted PCOS risk score is <b style="color:#7c3aed">{prob*100:.1f}%</b>.
                    </p>
                </div>""", unsafe_allow_html=True)

                advice = {
                    "Low":      "Your indicators are within a healthy range. Maintain a balanced diet and regular exercise. Consider annual gynaecological check-ups.",
                    "Moderate": "Some indicators suggest possible hormonal imbalance. Consider consulting a doctor for further evaluation and monitoring.",
                    "High":     "Multiple risk indicators are elevated. We strongly recommend consulting a gynaecologist or endocrinologist for proper diagnosis.",
                }
                icon_map = {"Low": "💚", "Moderate": "🟡", "High": "🔴"}
                st.markdown(f"""
                <div class="card">
                    <div class="card-title">{icon_map[level]} Recommendation</div>
                    <p style="font-size:14px">{advice[level]}</p>
                </div>""", unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("### 🧠 Explainable AI — Top Contributing Factors")
            xai_cols = st.columns(len(top_feats) if top_feats else 1)
            colors   = ["#ef4444", "#f97316", "#f59e0b", "#84cc16", "#10b981"]
            for i, feat in enumerate(top_feats[:5]):
                with xai_cols[i % len(xai_cols)]:
                    st.markdown(f"""
                    <div class="card" style="text-align:center;border-color:{colors[i]}">
                        <div style="font-size:24px">{'🔺' if i < 2 else '🔸'}</div>
                        <div style="color:{colors[i]};font-weight:700;margin-top:8px">#{i+1}</div>
                        <div style="color:#3b0764;font-size:13px;margin-top:4px">{feat}</div>
                    </div>""", unsafe_allow_html=True)

            with st.expander("📋 View Input Summary"):
                df_in = pd.DataFrame([{
                    "Feature": k.replace("_", " ").title(),
                    "Value":   v,
                    "Type":    "Numeric" if k in [x[1] for x in NUMERIC_FEATURES] else "Binary (Yes/No)"
                } for k, v in inputs.items()])
                st.dataframe(df_in, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════
#  PAGE 3 – CHATBOT
# ═══════════════════════════════════════════════
elif "Chatbot" in page:
    from src.chatbot import get_bot_response, analyze_sentiment
    import random
    import time

    # ── Daily Tips pool ──
    DAILY_TIPS = [
        "Even a 5% reduction in body weight can significantly improve PCOS symptoms and hormonal balance.",
        "A low-GI diet helps stabilize insulin levels — try swapping white rice for brown rice or quinoa.",
        "Regular exercise (even 30 min of walking daily) can reduce androgen levels in PCOS.",
        "Tracking your menstrual cycle with an app gives your doctor valuable diagnostic information.",
        "Stress raises cortisol, which can worsen PCOS — try 10 minutes of mindfulness daily.",
        "Vitamin D deficiency is common in PCOS — ask your doctor about checking your levels.",
        "Spearmint tea has shown mild anti-androgen effects in some PCOS studies.",
        "Getting 7–9 hours of sleep helps regulate the hormones affected by PCOS.",
        "Inositol supplements (especially Myo-inositol) may support insulin sensitivity in PCOS.",
        "PCOS is manageable — millions of people live healthy, full lives with the right support.",
    ]

    # ── FAQ data ──
    FAQ_LIST = [
        ("What is PCOS?",                    "what is pcos"),
        ("What are the symptoms?",           "what are pcos symptoms"),
        ("How is PCOS diagnosed?",           "how is pcos diagnosed"),
        ("What should I eat?",               "what should I eat for pcos"),
        ("How does exercise help?",          "how does exercise help pcos"),
        ("Can PCOS affect pregnancy?",       "pcos and pregnancy fertility"),
        ("How to manage weight with PCOS?",  "weight management with pcos"),
        ("What treatments are available?",   "what treatments are available for pcos"),
        ("How does stress affect PCOS?",     "how does stress affect pcos"),
        ("Explain my risk result",           "what does my risk result mean"),
    ]

    # ── Quick reply suggestions ──
    QUICK_REPLIES = [
        "What are PCOS symptoms?",
        "What should I eat?",
        "Explain my result",
        "Exercise tips for PCOS",
        "How is PCOS treated?",
        "PCOS and stress",
    ]

    # ── Header with illustration ──
    hdr_col, title_col = st.columns([1, 3])
    with hdr_col:
        st.markdown(f'<div class="illus-center" style="margin-top:10px">{ILLUS_CHAT}</div>',
                    unsafe_allow_html=True)
    with title_col:
        st.markdown("## 🤖 PCOS AI Chatbot")
        st.markdown("Ask anything about PCOS — symptoms, lifestyle, diet, or your results.")
        st.markdown("""
        <div style="background:#ede9fe;border-radius:10px;padding:10px 16px;margin-top:8px;border:1px solid #c4b5fd">
            <small style="color:#4c1d95">
            💡 The chatbot detects your mood and adjusts its tone automatically — try saying how you feel!
            </small>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    

    # ── Daily Tip of the Day ──
    if "daily_tip" not in st.session_state:
        st.session_state.daily_tip = random.choice(DAILY_TIPS)

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#ede9fe,#ddd6fe);
                border:1.5px solid #a78bfa;border-radius:12px;
                padding:14px 18px;margin-bottom:16px;">
        <span style="color:#7c3aed;font-weight:700;font-size:14px">💡 Tip of the Day</span><br>
        <span style="color:#3b0764;font-size:14px;line-height:1.6">{st.session_state.daily_tip}</span>
    </div>""", unsafe_allow_html=True)

    # ── Context banner ONLY if prediction was already done ──
    last = st.session_state.get("last_prediction")
    if last:
        level       = last["level"]
        score       = last["prob"] * 100
        badge_color = {"Low": "#065f46", "Moderate": "#92400e", "High": "#991b1b"}[level]
        bg_color    = {"Low": "#d1fae5", "Moderate": "#fef3c7", "High": "#fee2e2"}[level]
        border_col  = {"Low": "#6ee7b7", "Moderate": "#fcd34d", "High": "#fca5a5"}[level]
        st.markdown(f"""
        <div style="background:{bg_color};border:1.5px solid {border_col};border-radius:12px;
                    padding:12px 18px;margin-bottom:12px;">
            🔗 <b style="color:{badge_color}">Context loaded:</b>
            <span style="color:{badge_color}"> Your last risk result was
            <b>{level} Risk ({score:.1f}%)</b> — the chatbot will personalize responses accordingly.</span>
        </div>""", unsafe_allow_html=True)

        # ── Mood card based on sentiment of last user message ──
        if st.session_state.get("last_sentiment"):
            s = st.session_state.last_sentiment
            mood_map = {
                "Negative": ("😟", "#fff1f2", "#fca5a5", "#991b1b",
                             "You seem a little anxious or stressed. That's completely understandable with PCOS. "
                             "Remember — you're not alone, and managing PCOS is very possible with the right support. 💙"),
                "Positive": ("😊", "#f0fdf4", "#86efac", "#065f46",
                             "You're in a great headspace! Positive thinking really does support your health journey. Keep it up! 🌟"),
                "Neutral":  ("😐", "#f5f3ff", "#c4b5fd", "#4c1d95",
                             "Feeling neutral today? That's okay too. Ask me anything and I'm here to help."),
            }
            if s in mood_map:
                emoji, bg, border, txt, msg = mood_map[s]
                st.markdown(f"""
                <div style="background:{bg};border:1.5px solid {border};border-radius:12px;
                            padding:12px 18px;margin-bottom:12px;">
                    {emoji} <b style="color:{txt}">Mood Detected: {s}</b><br>
                    <span style="color:{txt};font-size:13px">{msg}</span>
                </div>""", unsafe_allow_html=True)

    # ── Init chat history ──
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [{
            "role": "bot",
            "text": "👋 Hello! I'm your PCOS awareness assistant. Ask me about symptoms, diet, lifestyle tips, or your recent risk result. How are you feeling today?",
        }]

    # ── Chat display ──
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"<div class='chat-user'>🧑 {msg['text']}</div>", unsafe_allow_html=True)
        elif msg.get("typing"):
            st.markdown("""
            <div class='chat-bot' style='font-style:italic;opacity:0.7'>
                🤖 &nbsp;<span>typing</span>
                <span style='animation:blink 1s infinite'>...</span>
            </div>
            <style>
                @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.2} }
            </style>""", unsafe_allow_html=True)
        else:
            cls = "chat-bot"
            if last:
                cls = {"Low": "chat-bot", "Moderate": "chat-bot-warn",
                       "High": "chat-bot-alert"}.get(last["level"], "chat-bot")
            st.markdown(f"<div class='{cls}'>🤖 {msg['text']}</div>", unsafe_allow_html=True)

    st.markdown("---")

    # ── Quick Reply Buttons ──
    st.markdown("<p style='color:#7c3aed;font-weight:600;font-size:13px;margin-bottom:6px'>⚡ Quick Replies:</p>",
                unsafe_allow_html=True)
    qr_cols = st.columns(len(QUICK_REPLIES))
    quick_input = None
    for i, qr in enumerate(QUICK_REPLIES):
        with qr_cols[i]:
            if st.button(qr, key=f"qr_{i}", help=f"Ask: {qr}"):
                quick_input = qr

    st.markdown("")

    # ── Text input + Send/Clear ──
    user_input = st.text_input("💬 Type your message...",
                               placeholder="e.g. What are common PCOS symptoms?",
                               key="chat_input")
    send_col, clear_col = st.columns([4, 1])
    with send_col:
        send = st.button("Send ➤")
    with clear_col:
        if st.button("🗑 Clear"):
            st.session_state.chat_history    = []
            st.session_state.last_sentiment  = None
            st.rerun()

    # ── FAQ Expander ──
    with st.expander("📖 Frequently Asked Questions — click to auto-ask"):
        st.markdown("<p style='color:#6b7280;font-size:13px'>Click any question to get an instant answer:</p>",
                    unsafe_allow_html=True)
        faq_cols = st.columns(2)
        for i, (question, trigger) in enumerate(FAQ_LIST):
            with faq_cols[i % 2]:
                if st.button(f"❓ {question}", key=f"faq_{i}"):
                    quick_input = trigger

    # ── Handle message sending (typed or quick reply or FAQ) ──
    final_input = None
    if send and user_input.strip():
        final_input = user_input.strip()
    elif quick_input:
        final_input = quick_input

    if final_input:
        sentiment = analyze_sentiment(final_input)
        st.session_state.last_sentiment = sentiment

        # Add user message
        st.session_state.chat_history.append({"role": "user", "text": final_input})

        # Show typing indicator briefly
        st.session_state.chat_history.append({"role": "bot", "typing": True, "text": ""})
        st.rerun()

    # ── Replace typing indicator with real response ──
    if st.session_state.get("chat_history") and st.session_state.chat_history[-1].get("typing"):
        time.sleep(0.8)
        # Find the user message before the typing indicator
        user_msg = ""
        for m in reversed(st.session_state.chat_history[:-1]):
            if m["role"] == "user":
                user_msg = m["text"]
                break
        sentiment = st.session_state.get("last_sentiment", "Neutral")
        response  = get_bot_response(user_msg, sentiment, last)
        # Replace typing with real response
        st.session_state.chat_history[-1] = {"role": "bot", "text": response}
        st.rerun()

# ═══════════════════════════════════════════════
#  PAGE 4 – MODEL INSIGHTS
# ═══════════════════════════════════════════════
elif "Insights" in page:
    # Header with illustration
    hdr_col, title_col = st.columns([1, 3])
    with hdr_col:
        st.markdown(f'<div class="illus-center" style="margin-top:10px">{ILLUS_CHART}</div>',
                    unsafe_allow_html=True)
    with title_col:
        st.markdown("## 📊 Model Insights & Performance")
        st.markdown("Detailed performance metrics for the selected model used in this application.")

    st.markdown("---")
    st.markdown("""
    <div class="card" style="border-color:#a78bfa;background:#faf5ff;">
        <div class="card-title">🏅 Model Chosen — XGBoost</div>
        <p style="font-size:15px;color:#3b0764;">
            <b style="color:#7c3aed;font-size:20px">XGBoost (Extreme Gradient Boosting)</b> was selected
            as the final model for this PCOS Risk Analyzer.
        </p>
        <p style="font-size:14px;color:#6b7280;margin-top:6px;">
            XGBoost is an ensemble learning algorithm that builds multiple decision trees sequentially,
            each correcting the errors of the previous. It is highly effective for medical classification
            tasks due to its ability to handle imbalanced data and capture complex feature interactions.
        </p>
    </div>""", unsafe_allow_html=True)

    st.markdown("### 📈 XGBoost Model Metrics")

    xgb_res = assets.get("xgb_results")

    def get_metric(res, *keys):
        if res:
            for k in keys:
                if k in res:
                    v = res[k]
                    return f"{float(v):.3f}" if isinstance(v, (int, float)) else str(v)
        return None

    metrics_display = {
        "Accuracy":  get_metric(xgb_res, "accuracy",  "Accuracy")  or f"{XGB_METRICS['Accuracy']:.3f}",
        "Precision": get_metric(xgb_res, "precision", "Precision") or f"{XGB_METRICS['Precision']:.3f}",
        "Recall":    get_metric(xgb_res, "recall",    "Recall")    or f"{XGB_METRICS['Recall']:.3f}",
        "F1-Score":  get_metric(xgb_res, "f1",        "F1-Score",  "f1_score") or f"{XGB_METRICS['F1-Score']:.3f}",
        "ROC-AUC":   get_metric(xgb_res, "roc_auc",   "ROC-AUC",  "AUC")      or f"{XGB_METRICS['ROC-AUC']:.3f}",
    }

    mcols = st.columns(5)
    metric_colors = ["#7c3aed", "#10b981", "#f59e0b", "#3b82f6", "#ec4899"]
    for col, (name, val), clr in zip(mcols, metrics_display.items(), metric_colors):
        with col:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-val" style="color:{clr}">{val}</div>
                <div class="metric-lbl">{name}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("")

    with st.expander("ℹ️ What do these metrics mean?"):
        st.markdown("""
        | Metric | What it means |
        |--------|--------------|
        | **Accuracy** | Out of all predictions, how many were correct (86.8%) |
        | **Precision** | Of all predicted PCOS cases, how many actually had PCOS (86.2%) |
        | **Recall** | Of all actual PCOS cases, how many did the model correctly catch (69.4%) |
        | **F1-Score** | Balance between Precision and Recall (76.9%) |
        | **ROC-AUC** | Overall ability to distinguish PCOS vs non-PCOS — higher is better (91.5%) |
        """)

    st.markdown("---")

    cmp = assets.get("comparison_df")
    if cmp is not None:
        st.markdown("### 🏆 All Models Comparison")
        st.dataframe(cmp, use_container_width=True, hide_index=True)
        st.markdown("")

    img_dir  = os.path.join(BASE, "models")
    img_tabs = st.tabs(["📉 ROC Curves", "🌟 Feature Importance", "📊 Performance Metrics"])

    for tab, fname in zip(img_tabs, ["roc_curves.png", "feature_importance.png", "performance_metrics.png"]):
        with tab:
            path = os.path.join(img_dir, fname)
            if os.path.exists(path):
                img = Image.open(path)
                st.image(img, width=900)
            else:
                st.info(f"Image `{fname}` not found in models/ folder.")

    st.markdown("### 📌 Feature Relevance Overview")
    feat_scores = {
        "Cycle Length (days)": 0.91,
        "BMI":                 0.85,
        "Hair Growth":         0.82,
        "Weight Gain":         0.78,
        "Skin Darkening":      0.74,
        "Pimples":             0.70,
        "Hair Loss":           0.65,
        "Fast Food":           0.58,
        "Weight (kg)":         0.55,
        "Age (years)":         0.50,
        "Regular Exercise":    0.45,
        "Height (cm)":         0.30,
    }
    df_feat = pd.DataFrame(feat_scores.items(), columns=["Feature", "Importance Score"])
    df_feat = df_feat.sort_values("Importance Score", ascending=True)
    fig_feat = px.bar(
        df_feat, x="Importance Score", y="Feature", orientation="h",
        color="Importance Score",
        color_continuous_scale=["#c4b5fd", "#8b5cf6", "#6d28d9", "#4c1d95"],
    )
    fig_feat.update_layout(
        paper_bgcolor="#ffffff", plot_bgcolor="#f9fafb",
        font_color="#3b0764", height=420,
        coloraxis_showscale=False,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(gridcolor="#e9d5ff"),
        yaxis=dict(gridcolor="#e9d5ff"),
    )
    st.plotly_chart(fig_feat, use_container_width=True)

# ═══════════════════════════════════════════════
#  PAGE 5 – ABOUT
# ═══════════════════════════════════════════════
elif "About" in page:
    # Header with illustration
    hdr_col, title_col = st.columns([1, 3])
    with hdr_col:
        st.markdown(f'<div class="illus-center" style="margin-top:10px">{ILLUS_HOME}</div>',
                    unsafe_allow_html=True)
    with title_col:
        st.markdown("## ℹ️ About This Project")
        st.markdown("An AI-powered PCOS awareness tool built with care by a two-person team.")

    st.markdown("---")

    st.markdown("""
    <div class="card">
        <div class="card-title">🎓 Project Overview</div>
        <p>This AI-powered PCOS Risk Analyzer was built as part of a collaborative academic project.
        It uses machine learning models trained on clinical PCOS data to provide early risk assessment.</p>
    </div>

    <div class="card">
        <div class="card-title">👩‍💻 Team Contributions</div>
        <p><b style="color:#7c3aed">🧠 Anjali — ML & Intelligence Lead</b><br>
        Data cleaning, feature selection, model training (LR, RF, XGB), hyperparameter tuning,
        model evaluation, risk thresholding, and model artifacts.</p>
        <br>
        <p><b style="color:#059669">💻 Meihul — System & Integration Lead</b><br>
        Streamlit app, UI/UX design, rule-based chatbot, VADER sentiment detection,
        explainable AI section, model insights visualization, sidebar navigation.</p>
    </div>

    <div class="card">
        <div class="card-title">🛠 Tech Stack</div>
        <p>Python • Streamlit • Scikit-learn • XGBoost • Plotly • VADER (NLTK) • Pickle • Pandas • NumPy</p>
    </div>

    <div class="card">
        <div class="card-title">📊 Model Selected</div>
        <p><b style="color:#7c3aed">XGBoost</b> (Extreme Gradient Boosting) was chosen as the final model
        based on its strong overall performance across all evaluation metrics.<br><br>
        Other models trained: Logistic Regression • Random Forest</p>
    </div>

    <div class="card" style="border-color:#fca5a5">
        <div class="card-title" style="color:#dc2626">⚠️ Disclaimer</div>
        <p>This tool is for <b>educational and awareness purposes only</b>. It does not provide
        medical diagnosis. Please consult a qualified healthcare professional for any medical advice.</p>
    </div>
    """, unsafe_allow_html=True)