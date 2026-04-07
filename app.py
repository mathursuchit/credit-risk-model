import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Credit Risk Predictor", page_icon="🏦", layout="centered")

st.title("🏦 Credit Risk Predictor")
st.markdown("Enter applicant details to predict credit risk using an XGBoost model trained on the German Credit dataset.")

@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    features = joblib.load("feature_names.pkl")
    return model, features

try:
    model, feature_names = load_model()
except FileNotFoundError:
    st.error("Model not found. Please run the notebook first to train and save the model.")
    st.stop()

st.subheader("Applicant Information")

col1, col2 = st.columns(2)

with col1:
    duration = st.slider("Loan Duration (months)", 6, 72, 24)
    credit_amount = st.number_input("Credit Amount ($)", 500, 20000, 5000, step=500)
    age = st.slider("Age", 18, 75, 35)
    installment_commitment = st.slider("Installment Rate (% of income)", 1, 4, 2)

with col2:
    residence_since = st.slider("Years at Current Residence", 1, 4, 2)
    existing_credits = st.slider("Number of Existing Credits", 1, 4, 1)
    num_dependents = st.slider("Number of Dependents", 1, 2, 1)

# Build input — fill non-UI features with median values
input_data = {}
medians = {
    'duration': 24, 'credit_amount': 2320, 'installment_commitment': 2,
    'residence_since': 3, 'age': 35, 'existing_credits': 1, 'num_dependents': 1,
    'checking_status': 1, 'credit_history': 2, 'purpose': 3, 'savings_status': 1,
    'employment': 2, 'personal_status': 2, 'other_parties': 0, 'property_magnitude': 2,
    'other_payment_plans': 0, 'housing': 1, 'job': 2, 'own_telephone': 0,
    'foreign_worker': 1
}

for feat in feature_names:
    if feat == 'duration':
        input_data[feat] = duration
    elif feat == 'credit_amount':
        input_data[feat] = credit_amount
    elif feat == 'age':
        input_data[feat] = age
    elif feat == 'installment_commitment':
        input_data[feat] = installment_commitment
    elif feat == 'residence_since':
        input_data[feat] = residence_since
    elif feat == 'existing_credits':
        input_data[feat] = existing_credits
    elif feat == 'num_dependents':
        input_data[feat] = num_dependents
    else:
        input_data[feat] = medians.get(feat, 1)

input_df = pd.DataFrame([input_data])

if st.button("Predict Credit Risk", type="primary"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    st.divider()
    col1, col2, col3 = st.columns(3)

    with col1:
        label = "Bad Risk" if prediction == 1 else "Good Risk"
        color = "🔴" if prediction == 1 else "🟢"
        st.metric("Prediction", f"{color} {label}")

    with col2:
        st.metric("Good Credit Probability", f"{probability[0]*100:.1f}%")

    with col3:
        st.metric("Bad Credit Probability", f"{probability[1]*100:.1f}%")

    st.progress(float(probability[0]), text=f"Confidence: {max(probability)*100:.1f}%")

    if prediction == 1:
        st.warning("**High Risk Applicant.** Recommend additional review before approval.")
    else:
        st.success("**Low Risk Applicant.** Meets standard credit criteria.")

    with st.expander("Feature Importances"):
        importances = pd.Series(
            model.feature_importances_, index=feature_names
        ).sort_values(ascending=False)
        st.bar_chart(importances)

st.divider()
st.caption("Model: XGBoost tuned with GridSearchCV | Dataset: German Credit (UCI/OpenML) | Author: Suchit Mathur")
