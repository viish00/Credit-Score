import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn

st.set_page_config(page_title="Credit Risk AI", page_icon="📊", layout="centered")

class CreditRiskANN(nn.Module):
    def __init__(self, input_features):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_features, 64),
            nn.ReLU(),
            nn.Dropout(p=0.20),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p=0.20),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)

@st.cache_resource
def load_assets():
    required = [
        "credit_risk_ann.pth",
        "credit_risk_scaler.joblib",
        "credit_risk_feature_names.joblib"
    ]
    missing = [f for f in required if not os.path.exists(f)]
    if missing:
        raise FileNotFoundError(
            "Missing model files: " + ", ".join(missing) +
            ". Add the three trained artifacts to the project root."
        )

    feature_names = joblib.load("credit_risk_feature_names.joblib")
    scaler = joblib.load("credit_risk_scaler.joblib")
    model = CreditRiskANN(input_features=len(feature_names))
    state = torch.load("credit_risk_ann.pth", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, scaler, feature_names

st.title("Credit Risk AI Analysis")
st.caption("Artificial Neural Network-based credit default risk estimation")
st.info("Educational project using the German Credit dataset. This result is not a real lending decision or financial advice.")

st.subheader("Applicant Details")
with st.form("credit_application"):
    c1, c2 = st.columns(2)
    with c1:
        status = st.selectbox("Checking account status", ["A11", "A12", "A13", "A14"])
        duration = st.number_input("Loan duration (months)", min_value=4, max_value=72, value=12)
        credit_history = st.selectbox("Credit history", ["A30", "A31", "A32", "A33", "A34"])
        purpose = st.selectbox("Loan purpose", ["A40", "A41", "A410", "A42", "A43", "A44", "A45", "A46", "A48", "A49"])
        credit_amount = st.number_input("Credit amount", min_value=250, max_value=20000, value=3000, step=100)
        savings = st.selectbox("Savings account / bonds", ["A61", "A62", "A63", "A64", "A65"])
        employment = st.selectbox("Present employment", ["A71", "A72", "A73", "A74", "A75"])
        installment_rate = st.selectbox("Installment rate (% of disposable income)", [1, 2, 3, 4])
        personal_status = st.selectbox("Personal status and sex code", ["A91", "A92", "A93", "A94", "A95"])
        other_debtors = st.selectbox("Other debtors / guarantors", ["A101", "A102", "A103"])
    with c2:
        residence_since = st.selectbox("Present residence since (years)", [1, 2, 3, 4])
        property_code = st.selectbox("Property", ["A121", "A122", "A123", "A124"])
        age = st.number_input("Age (years)", min_value=18, max_value=100, value=30)
        other_installment_plans = st.selectbox("Other installment plans", ["A141", "A142", "A143"])
        housing = st.selectbox("Housing", ["A151", "A152", "A153"])
        existing_credits = st.selectbox("Existing credits at this bank", [1, 2, 3, 4])
        job = st.selectbox("Job category", ["A171", "A172", "A173", "A174"])
        dependents = st.selectbox("Number of people liable for maintenance", [1, 2])
        telephone = st.selectbox("Telephone", ["A191", "A192"])
        foreign_worker = st.selectbox("Foreign worker", ["A201", "A202"])

    submitted = st.form_submit_button("Predict Credit Risk", use_container_width=True)

if submitted:
    try:
        model, scaler, feature_names = load_assets()

        row = pd.DataFrame([{
            "Status": status,
            "Duration": duration,
            "CreditHistory": credit_history,
            "Purpose": purpose,
            "CreditAmount": credit_amount,
            "Savings": savings,
            "Employment": employment,
            "InstallmentRate": installment_rate,
            "PersonalStatus": personal_status,
            "OtherDebtors": other_debtors,
            "ResidenceSince": residence_since,
            "Property": property_code,
            "Age": age,
            "OtherInstallmentPlans": other_installment_plans,
            "Housing": housing,
            "ExistingCredits": existing_credits,
            "Job": job,
            "Dependents": dependents,
            "Telephone": telephone,
            "ForeignWorker": foreign_worker
        }])

        categorical_cols = row.select_dtypes(include="object").columns.tolist()
        encoded = pd.get_dummies(row, columns=categorical_cols, drop_first=True, dtype=float)
        encoded = encoded.reindex(columns=feature_names, fill_value=0)
        scaled = scaler.transform(encoded)

        with torch.no_grad():
            x = torch.tensor(np.asarray(scaled), dtype=torch.float32)
            logit = model(x)
            probability = torch.sigmoid(logit).item()

        # Match the notebook's default 0.50 classification threshold.
        threshold = 0.50
        if probability >= threshold:
            st.error("Prediction: Higher Default Risk")
        else:
            st.success("Prediction: Lower Default Risk")

        st.metric("Estimated default probability", f"{probability * 100:.2f}%")
        st.progress(float(np.clip(probability, 0, 1)))
        st.caption("Classification threshold: 50%. Model probabilities have not been independently calibrated.")
    except Exception as e:
        st.error(f"Could not load or run the model: {e}")

st.divider()
st.caption("Project by Vishal Ranwa and Krish Gupta | Fundamentals of Deep Learning")
