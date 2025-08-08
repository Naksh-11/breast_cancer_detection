import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.datasets import load_breast_cancer

# Load saved models & scaler
lr = joblib.load("logistic_regression_model.pkl")
svm_model = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")

# Dataset for feature names
data = load_breast_cancer()

st.title("🩺 Breast Cancer Detection App")
st.write("Enter patient measurements to predict breast cancer.")

# Input fields
input_data = []
for feature in data.feature_names:
    input_data.append(st.number_input(feature, min_value=0.0, value=float(np.mean(data.data[:, list(data.feature_names).index(feature)]))))

# Model choice
model_choice = st.selectbox("Choose Model", ["Logistic Regression", "SVM"])

if st.button("Predict"):
    input_array = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_array)

    if model_choice == "Logistic Regression":
        prediction = lr.predict(input_scaled)[0]
    else:
        prediction = svm_model.predict(input_scaled)[0]

    if prediction == 1:
        st.success("Prediction: ✅ Benign (Not Cancerous)")
    else:
        st.error("Prediction: ⚠️ Malignant (Cancerous)")
