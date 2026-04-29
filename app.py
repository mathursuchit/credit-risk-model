import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Credit Risk Scorer", page_icon=None, layout="centered")

st.title("Credit Risk Scorer")
st.markdown("Enter applicant details to predict their credit risk tier using an XGBoost model trained on 51,336 real credit bureau applications.")

TIER_CONFIG = {
    'P1': {'label': 'P1 — Low Risk',        'color': 'green',  'advice': 'Strong credit profile. Approve with standard terms.'},
    'P2': {'label': 'P2 — Moderate Risk',   'color': 'blue',   'advice': 'Acceptable profile. Consider standard or slightly adjusted terms.'},
    'P3': {'label': 'P3 — Elevated Risk',   'color': 'orange', 'advice': 'Higher risk profile. Review manually before approving.'},
    'P4': {'label': 'P4 — High Risk',       'color': 'red',    'advice': 'High default risk. Recommend rejection or collateral requirement.'},
}

@st.cache_resource
def load_model():
    model   = joblib.load("model.pkl")
    features = joblib.load("feature_names.pkl")
    le      = joblib.load("label_encoder.pkl")
    return model, features, le

try:
    model, feature_names, le = load_model()
except FileNotFoundError:
    st.error("Model not found. Please run notebook.ipynb first to train and save the model.")
    st.stop()

st.subheader("Applicant Information")

col1, col2 = st.columns(2)

with col1:
    age         = st.slider("Age", 21, 70, 35)
    income      = st.number_input("Net Monthly Income", 5000, 500000, 50000, step=5000)
    employment  = st.slider("Years with Current Employer", 0, 20, 3)
    education   = st.selectbox("Education Level",
                    ["SSC (10th)", "12th", "Graduate", "Post-Graduate", "Professional"])

with col2:
    marital     = st.selectbox("Marital Status", ["Married", "Single", "Divorced", "Widowed"])
    gender      = st.selectbox("Gender", ["Male", "Female"])
    missed_pmnt = st.slider("Total Missed Payments", 0, 20, 0)
    delinquent  = st.slider("Number of Times Delinquent", 0, 10, 0)

edu_map = {
    "SSC (10th)": 1, "12th": 2, "Graduate": 3,
    "Post-Graduate": 4, "Professional": 3
}

# Build input with user values for known fields, medians for the rest
input_data = {}
for feat in feature_names:
    if feat == 'AGE':
        input_data[feat] = age
    elif feat == 'NETMONTHLYINCOME':
        input_data[feat] = income
    elif feat == 'Time_With_Curr_Empr':
        input_data[feat] = employment
    elif feat == 'EDUCATION':
        input_data[feat] = edu_map[education]
    elif feat == 'Tot_Missed_Pmnt':
        input_data[feat] = missed_pmnt
    elif feat == 'num_times_delinquent':
        input_data[feat] = delinquent
    elif 'MARITALSTATUS' in feat:
        val = marital.upper()
        input_data[feat] = 1 if val in feat.upper() else 0
    elif 'GENDER' in feat:
        input_data[feat] = 1 if gender.upper() in feat.upper() else 0
    else:
        input_data[feat] = 0

input_df = pd.DataFrame([input_data])

if st.button("Predict Credit Tier", type="primary"):
    pred_encoded = model.predict(input_df)[0]
    pred_label   = le.inverse_transform([pred_encoded])[0]
    probabilities = model.predict_proba(input_df)[0]

    tier = TIER_CONFIG[pred_label]

    st.divider()
    st.subheader(tier['label'])
    st.write(f"**Recommendation:** {tier['advice']}")

    st.divider()
    st.subheader("Probability by Tier")
    prob_df = pd.DataFrame({
        'Tier': le.classes_,
        'Probability': [f"{p*100:.1f}%" for p in probabilities],
        'Score': probabilities
    }).set_index('Tier')

    col1, col2, col3, col4 = st.columns(4)
    cols = [col1, col2, col3, col4]
    for i, (tier_name, row) in enumerate(prob_df.iterrows()):
        cols[i].metric(tier_name, f"{row['Score']*100:.1f}%")

    st.bar_chart(pd.Series(probabilities, index=le.classes_))

    with st.expander("Top Feature Importances"):
        importances = pd.Series(
            model.feature_importances_, index=feature_names
        ).nlargest(15).sort_values()
        st.bar_chart(importances)

st.divider()
st.caption("Model: XGBoost tuned with GridSearchCV | Dataset: 51,336 credit bureau applications | Tiers: P1 (low risk) → P4 (high risk) | Author: Suchit Mathur")
