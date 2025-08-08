import streamlit as st
import numpy as np
import joblib
from sklearn.datasets import load_breast_cancer

# Load models & scaler
lr = joblib.load("logistic_regression_model.pkl")
svm_model = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")
data = load_breast_cancer()

# Page config
st.set_page_config(page_title="Breast Cancer Detection", page_icon="🩺", layout="wide")

# Header
st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>🩺 Breast Cancer Detection</h1>", unsafe_allow_html=True)
st.write("---")

# Create two columns
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Enter Patient Data")
    input_data = []
    for feature in data.feature_names:
        val = st.number_input(f"{feature}", min_value=0.0,
                              value=float(np.mean(data.data[:, list(data.feature_names).index(feature)])))
        input_data.append(val)

with col2:
    st.subheader("Model Selection")
    model_choice = st.selectbox("Choose a model", ["Logistic Regression", "SVM"])

    if st.button("Predict", use_container_width=True):
        input_array = np.array(input_data).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        if model_choice == "Logistic Regression":
            prediction = lr.predict(input_scaled)[0]
        else:
            prediction = svm_model.predict(input_scaled)[0]

        st.write("### Result:")
        if prediction == 1:
            st.success("✅ Benign (Not Cancerous)")
            st.balloons()
        else:
            st.error("⚠ Malignant (Cancerous)")
