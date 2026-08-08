import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

st.set_page_config(page_title="Heart Disease Risk Predictor", page_icon="❤️", layout="centered")

# ---------------- Load model artifact ----------------
@st.cache_resource
def load_artifact():
    return joblib.load("heart_disease_model.pkl")

try:
    artifact = load_artifact()
    model = artifact["model"]
    scaler = artifact["scaler"]
    selected_features = artifact["selected_features"]
    continuous_cols = artifact["continuous_cols"]
    all_columns = artifact["all_columns"]
    categorical_cols = artifact["categorical_cols"]
except FileNotFoundError:
    st.error("`heart_disease_model.pkl` not found. Please download it from the Colab notebook and place it in this folder.")
    st.stop()

st.title("❤️ Heart Disease Risk Predictor")
st.caption("Logistic Regression model — trained on the UCI Cleveland Heart Disease dataset")

st.markdown("---")
st.subheader("Enter Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=50)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", [
        "Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"
    ])
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=220, value=120)
    chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", ["No", "Yes"])
    restecg = st.selectbox("Resting ECG Result", ["Normal", "ST-T abnormality", "LV hypertrophy"])

with col2:
    thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
    exang = st.selectbox("Exercise-induced Angina?", ["No", "Yes"])
    oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=7.0, value=1.0, step=0.1)
    slope = st.selectbox("Slope of Peak Exercise ST", ["Upsloping", "Flat", "Downsloping"])
    ca = st.number_input("Number of Major Vessels (0-3)", min_value=0, max_value=3, value=0)
    thal = st.selectbox("Thalassemia", ["Normal", "Fixed defect", "Reversible defect"])

predict_btn = st.button("Predict Risk", use_container_width=True, type="primary")

if predict_btn:
    # ---------------- Build raw input row ----------------
    raw = {
        "age": age,
        "sex": 1 if sex == "Male" else 0,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": 1 if fbs == "Yes" else 0,
        "thalach": thalach,
        "exang": 1 if exang == "Yes" else 0,
        "oldpeak": oldpeak,
        "ca": ca,
        "cp": {"Typical angina": 1, "Atypical angina": 2, "Non-anginal pain": 3, "Asymptomatic": 4}[cp],
        "restecg": {"Normal": 0, "ST-T abnormality": 1, "LV hypertrophy": 2}[restecg],
        "slope": {"Upsloping": 1, "Flat": 2, "Downsloping": 3}[slope],
        "thal": {"Normal": 3, "Fixed defect": 6, "Reversible defect": 7}[thal],
    }

    row_df = pd.DataFrame([raw])

    # one-hot encode categorical to match training columns
    row_enc = pd.get_dummies(row_df, columns=[c for c in categorical_cols if c in row_df.columns])

    # align to training columns (fill missing dummy cols with 0)
    for col in all_columns:
        if col not in row_enc.columns:
            row_enc[col] = 0
    row_enc = row_enc[all_columns]

    # scale continuous columns
    row_enc[continuous_cols] = scaler.transform(row_enc[continuous_cols])

    # select final features used by the model
    row_final = row_enc[selected_features]

    prob = model.predict_proba(row_final)[0, 1]
    pred = int(prob >= 0.5)

    st.markdown("---")
    st.subheader("Result")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        title={"text": "Heart Disease Risk (%)"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "darkred" if pred == 1 else "green"},
            "steps": [
                {"range": [0, 40], "color": "#d4edda"},
                {"range": [40, 70], "color": "#fff3cd"},
                {"range": [70, 100], "color": "#f8d7da"},
            ],
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

    if pred == 1:
        st.error(f"High risk detected Probability: {prob:.1%}")
    else:
        st.success(f"Low risk Probability: {prob:.1%}")

    st.caption(
        "This model is for academic demo purposes only and is not a diagnostic tool. "
        "Please consult a doctor for actual medical decisions."
    )

# st.markdown("---")
# st.caption("CSE — AI/ML Course Project | Logistic Regression | UCI Cleveland Heart Disease Dataset")