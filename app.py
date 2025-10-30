# app.py
import streamlit as st
import pickle
import numpy as np
import pandas as pd

st.set_page_config(page_title="💳 Credit Analyzer", layout="centered")

# -------------------------------
# Load Trained Model
# -------------------------------
@st.cache_resource
def load_model():
    with open("credit_analyzer_model.pkl", "rb") as f:
        data = pickle.load(f)
    return data

model_data = load_model()
model = model_data["model"]
scaler = model_data["scaler"]
label_encoders = model_data["label_encoders"]
features = model_data["features"]

# -------------------------------
# App Header
# -------------------------------
st.title("💳 Credit Analyzer - Loan Default Predictor")
st.markdown("Predict whether a person will default on a loan using financial and personal information.")

st.divider()

# -------------------------------
# User Input Form
# -------------------------------
st.subheader("Enter Applicant Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    income = st.number_input("Monthly Income ($)", min_value=0, value=4000)
    loan_amount = st.number_input("Loan Amount ($)", min_value=0, value=10000)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=700)
    months_employed = st.number_input("Months Employed", min_value=0, value=24)
    num_credit_lines = st.number_input("Number of Credit Lines", min_value=0, value=3)

with col2:
    interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, value=5.0)
    loan_term = st.number_input("Loan Term (months)", min_value=6, max_value=360, value=60)
    dti_ratio = st.number_input("Debt-to-Income Ratio", min_value=0.0, value=0.3)
    education = st.selectbox("Education", ["High School", "Bachelors", "Masters", "PhD", "Other"])
    employment_type = st.selectbox("Employment Type", ["Salaried", "Self-Employed", "Unemployed", "Retired"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])

st.divider()

col3, col4, col5 = st.columns(3)

with col3:
    has_mortgage = st.selectbox("Has Mortgage?", ["No", "Yes"])
with col4:
    has_dependents = st.selectbox("Has Dependents?", ["No", "Yes"])
with col5:
    has_cosigner = st.selectbox("Has Co-signer?", ["No", "Yes"])

loan_purpose = st.selectbox(
    "Loan Purpose",
    ["Home", "Car", "Education", "Business", "Personal", "Other"]
)

# -------------------------------
# Prepare Data
# -------------------------------
input_dict = {
    "Age": age,
    "Income": income,
    "LoanAmount": loan_amount,
    "CreditScore": credit_score,
    "MonthsEmployed": months_employed,
    "NumCreditLines": num_credit_lines,
    "InterestRate": interest_rate,
    "LoanTerm": loan_term,
    "DTIRatio": dti_ratio,
    "Education": education,
    "EmploymentType": employment_type,
    "MaritalStatus": marital_status,
    "HasMortgage": has_mortgage,
    "HasDependents": has_dependents,
    "LoanPurpose": loan_purpose,
    "HasCoSigner": has_cosigner
}

input_df = pd.DataFrame([input_dict])

# Apply label encoders to categorical columns
# Apply label encoders to categorical columns (handle unseen labels safely)
for col, le in label_encoders.items():
    if col in input_df.columns:
        input_df[col] = input_df[col].apply(
            lambda x: le.transform([x])[0] if x in le.classes_ else -1  # unseen -> -1
        )


# Scale numerical features
scaled_input = scaler.transform(input_df[features])

# -------------------------------
# Prediction
# -------------------------------
if st.button("🔮 Predict Default Risk"):
    pred = model.predict(scaled_input)[0]
    prob = model.predict_proba(scaled_input)[0][1]  # probability of default

    st.subheader("📊 Prediction Result")
    if pred == 1:
        st.error(f"⚠️ High Risk of Default (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Low Risk of Default (Probability: {prob:.2f})")

    st.markdown("---")
    st.caption("Model trained on Loan Default Prediction dataset using Random Forest Classifier.")
