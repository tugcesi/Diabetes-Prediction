"""
🩺 Diabetes Risk Prediction
Model: VotingClassifier (XGBoost + LightGBM)
Files: model.joblib, scaler.joblib
"""

from __future__ import annotations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="🩺 Diabetes Risk Prediction",
    page_icon="🩺",
    layout="wide",
)

# ─── Paths ────────────────────────────────────────────────────────────────────
MODEL_PATH  = Path("model.joblib")
SCALER_PATH = Path("scaler.joblib")

# ─── Constants ────────────────────────────────────────────────────────────────
BMI_MIN = 15.1
BMI_MAX = 38.4
WHR_MIN = 0.68
WHR_MAX = 1.05

ETHNICITY_OPTIONS = ["Asian", "Black", "Hispanic", "Other", "White"]

# Columns after pd.get_dummies(drop_first=True)
# ethnicity   : drop 'Asian'     → Black, Hispanic, Other, White
# activity_category : drop 'Sedentary' → Low, Moderate, High
# bp_category : drop 'Normal'    → Elevated, High1, High2
FEATURE_COLS = [
    "family_history_diabetes",
    "age_bmi_interaction",
    "physical_activity_minutes_per_week",
    "age",
    "bmi_bp_interaction",
    "systolic_bp",
    "bmi",
    "ldl_cholesterol",
    "age_activity_interaction",
    "metabolic_score",
    "trig_hdl_ratio",
    "ldl_hdl_ratio",
    "triglycerides",
    "cholesterol_total",
    "waist_to_hip_ratio",
    "ethnicity_Black",
    "ethnicity_Hispanic",
    "ethnicity_Other",
    "ethnicity_White",
    "activity_category_Low",
    "activity_category_Moderate",
    "activity_category_High",
    "bp_category_Elevated",
    "bp_category_High1",
    "bp_category_High2",
]


# ─── Load artifacts ───────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        return None, None
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


# ─── Feature engineering ──────────────────────────────────────────────────────
def compute_features(p: dict) -> np.ndarray:
    age      = float(p["age"])
    bmi      = float(p["bmi"])
    whr      = float(p["waist_to_hip_ratio"])
    activity = float(p["physical_activity_minutes_per_week"])
    sbp      = float(p["systolic_bp"])
    hdl      = float(p["hdl_cholesterol"])
    ldl      = float(p["ldl_cholesterol"])
    trig     = float(p["triglycerides"])

    age_bmi_interaction       = age * bmi / 100
    age_activity_interaction  = age * (activity / 100)
    bmi_bp_interaction        = (bmi / 25) * (sbp / 120)
    metabolic_score           = (
        (bmi - BMI_MIN) / (BMI_MAX - BMI_MIN) * 50
        + (whr - WHR_MIN) / (WHR_MAX - WHR_MIN) * 50
    )
    ldl_hdl_ratio  = ldl / (hdl + 1)
    trig_hdl_ratio = trig / (hdl + 1)

    # bp_category
    if sbp <= 120:
        bp_cat = "Normal"
    elif sbp <= 130:
        bp_cat = "Elevated"
    elif sbp <= 140:
        bp_cat = "High1"
    else:
        bp_cat = "High2"

    # activity_category
    if activity <= 30:
        act_cat = "Sedentary"
    elif activity <= 75:
        act_cat = "Low"
    elif activity <= 150:
        act_cat = "Moderate"
    else:
        act_cat = "High"

    ethnicity = p["ethnicity"]

    row = {col: 0.0 for col in FEATURE_COLS}
    row["family_history_diabetes"]            = float(p["family_history_diabetes"])
    row["age_bmi_interaction"]                = age_bmi_interaction
    row["physical_activity_minutes_per_week"] = activity
    row["age"]                                = age
    row["bmi_bp_interaction"]                 = bmi_bp_interaction
    row["systolic_bp"]                        = sbp
    row["bmi"]                                = bmi
    row["ldl_cholesterol"]                    = ldl
    row["age_activity_interaction"]           = age_activity_interaction
    row["metabolic_score"]                    = metabolic_score
    row["trig_hdl_ratio"]                     = trig_hdl_ratio
    row["ldl_hdl_ratio"]                      = ldl_hdl_ratio
    row["triglycerides"]                      = trig
    row["cholesterol_total"]                  = float(p["cholesterol_total"])
    row["waist_to_hip_ratio"]                 = whr

    # Ethnicity dummies (drop: Asian)
    if ethnicity != "Asian":
        key = f"ethnicity_{ethnicity}"
        if key in row:
            row[key] = 1.0

    # Activity category dummies (drop: Sedentary)
    if act_cat != "Sedentary":
        key = f"activity_category_{act_cat}"
        if key in row:
            row[key] = 1.0

    # BP category dummies (drop: Normal)
    if bp_cat != "Normal":
        key = f"bp_category_{bp_cat}"
        if key in row:
            row[key] = 1.0

    return np.array([row[col] for col in FEATURE_COLS], dtype=np.float32).reshape(1, -1)


# ─── Gauge chart ──────────────────────────────────────────────────────────────
def create_gauge(prob: float) -> go.Figure:
    pct = prob * 100
    if pct < 30:
        bar_color = "#52c41a"
    elif pct < 60:
        bar_color = "#faad14"
    else:
        bar_color = "#f5222d"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix": "%", "font": {"size": 40}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar":  {"color": bar_color, "thickness": 0.3},
            "steps": [
                {"range": [0,  30], "color": "#f6ffed"},
                {"range": [30, 60], "color": "#fffbe6"},
                {"range": [60, 100], "color": "#fff1f0"},
            ],
            "threshold": {
                "line": {"color": bar_color, "width": 4},
                "thickness": 0.75,
                "value": pct,
            },
        },
        title={"text": "Diabetes Risk", "font": {"size": 18}},
    ))
    fig.update_layout(height=280, margin=dict(t=60, b=0, l=30, r=30))
    return fig


# ─── Risk banner ──────────────────────────────────────────────────────────────
def risk_banner(prob: float) -> None:
    pct = prob * 100
    if pct < 30:
        st.success(f"✅ **Düşük Risk** — Diyabet riski düşük görünüyor ({pct:.1f}%)")
    elif pct < 60:
        st.warning(f"⚠️ **Orta Risk** — Diyabet riski orta seviyede ({pct:.1f}%)")
    else:
        st.error(f"🚨 **Yüksek Risk** — Diyabet riski yüksek ({pct:.1f}%)")


# ─── Sidebar ──────────────────────────────────────────────────────────────────
def build_sidebar() -> dict:
    with st.sidebar:
        st.title("🩺 Diabetes Risk Predictor")
        st.markdown("---")
        st.markdown("### 👤 Kişisel Bilgiler")
        age            = st.slider("Age", 18, 90, 40)
        ethnicity      = st.selectbox("Ethnicity", ETHNICITY_OPTIONS)
        family_history = st.selectbox("Family History of Diabetes", ["No", "Yes"])

        st.markdown("### ⚖️ Vücut Ölçüleri")
        bmi          = st.slider("BMI", 15.1, 38.4, 25.0, 0.1)
        waist_to_hip = st.slider("Waist-to-Hip Ratio", 0.68, 1.05, 0.85, 0.01)

        st.markdown("### 🫀 Kan Basıncı & Kolesterol")
        systolic_bp       = st.slider("Systolic Blood Pressure (mmHg)", 90, 200, 120)
        hdl_cholesterol   = st.slider("HDL Cholesterol (mg/dL)", 20, 100, 50)
        ldl_cholesterol   = st.slider("LDL Cholesterol (mg/dL)", 50, 250, 120)
        triglycerides     = st.slider("Triglycerides (mg/dL)", 50, 500, 150)
        cholesterol_total = st.slider("Total Cholesterol (mg/dL)", 100, 400, 200)

        st.markdown("### 🏃 Aktivite")
        activity_minutes = st.slider("Physical Activity (min/week)", 0, 750, 100)

        st.markdown("---")
        predict_clicked = st.button(
            "🔍 Predict Risk", use_container_width=True, type="primary"
        )

    return {
        "payload": {
            "age":                                age,
            "ethnicity":                          ethnicity,
            "family_history_diabetes":            1 if family_history == "Yes" else 0,
            "bmi":                                bmi,
            "waist_to_hip_ratio":                 waist_to_hip,
            "systolic_bp":                        systolic_bp,
            "hdl_cholesterol":                    hdl_cholesterol,
            "ldl_cholesterol":                    ldl_cholesterol,
            "triglycerides":                      triglycerides,
            "cholesterol_total":                  cholesterol_total,
            "physical_activity_minutes_per_week": activity_minutes,
        },
        "predict_clicked": predict_clicked,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    model, scaler = load_artifacts()

    sidebar         = build_sidebar()
    payload         = sidebar["payload"]
    predict_clicked = sidebar["predict_clicked"]

    if model is None or scaler is None:
        st.error(
            "⚠️ **Model dosyaları bulunamadı!**\n\n"
            "`model.joblib` ve `scaler.joblib` dosyalarının "
            "repo kök dizininde mevcut olduğundan emin ol."
        )
        st.stop()

    features        = compute_features(payload)
    features_scaled = scaler.transform(features)
    prob            = float(model.predict_proba(features_scaled)[0][1])
    pred            = int(model.predict(features_scaled)[0])

    st.title("🩺 Diabetes Risk Prediction")
    st.markdown("Soldaki panelden bilgileri gir, anlık diyabet risk skorunu gör.")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🎯 Risk Score",        f"{prob * 100:.1f}%")
    c2.metric("📋 Prediction",        "Diabetic" if pred == 1 else "Non-Diabetic")
    c3.metric("⚖️ BMI",               f"{payload['bmi']}")
    c4.metric("🏃 Activity (min/wk)", f"{payload['physical_activity_minutes_per_week']}")

    if predict_clicked:
        st.markdown("---")
        risk_banner(prob)

    st.markdown("---")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.plotly_chart(create_gauge(prob), use_container_width=True)

    with col_right:
        st.markdown("#### 📋 Girilen Bilgiler")
        fh = "Var" if payload["family_history_diabetes"] == 1 else "Yok"
        st.dataframe(
            pd.DataFrame({
                "Özellik": [
                    "Yaş", "Etnisite", "Aile Geçmişi", "BMI",
                    "Bel/Kalça Oranı", "Sistolik KB", "HDL", "LDL",
                    "Trigliserit", "Toplam Kolesterol", "Aktivite (dk/hafta)",
                ],
                "Değer": [
                    payload["age"],
                    payload["ethnicity"],
                    fh,
                    payload["bmi"],
                    payload["waist_to_hip_ratio"],
                    payload["systolic_bp"],
                    payload["hdl_cholesterol"],
                    payload["ldl_cholesterol"],
                    payload["triglycerides"],
                    payload["cholesterol_total"],
                    payload["physical_activity_minutes_per_week"],
                ],
            }),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("---")
    st.markdown("#### ⚙️ Hesaplanan Özellikler")
    age      = payload["age"]
    bmi      = payload["bmi"]
    whr      = payload["waist_to_hip_ratio"]
    activity = payload["physical_activity_minutes_per_week"]
    sbp      = payload["systolic_bp"]
    hdl      = payload["hdl_cholesterol"]
    ldl      = payload["ldl_cholesterol"]
    trig     = payload["triglycerides"]
    ms = (
        (bmi - BMI_MIN) / (BMI_MAX - BMI_MIN) * 50
        + (whr - WHR_MIN) / (WHR_MAX - WHR_MIN) * 50
    )
    e1, e2, e3 = st.columns(3)
    e1.metric("Age × BMI",        f"{age * bmi / 100:.2f}")
    e2.metric("BMI × BP",         f"{(bmi / 25) * (sbp / 120):.2f}")
    e3.metric("Metabolic Score",  f"{ms:.1f}")
    e4, e5, e6 = st.columns(3)
    e4.metric("LDL/HDL Ratio",    f"{ldl / (hdl + 1):.2f}")
    e5.metric("Trig/HDL Ratio",   f"{trig / (hdl + 1):.2f}")
    e6.metric("Age × Activity",   f"{age * (activity / 100):.2f}")

    st.markdown("---")
    st.caption(
        "⚕️ Bu uygulama yalnızca bilgilendirme amaçlıdır, tıbbi teşhis değildir. | "
        "[GitHub](https://github.com/tugcesi/Diabetes-Prediction)"
    )


if __name__ == "__main__":
    main()
