import streamlit as st
import numpy as np
import joblib
import tensorflow as tf

# Load trained model and scaler
model = tf.keras.models.load_model("churn_model.h5")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊", layout="centered")

st.title("📊 Customer Churn Prediction App")
st.write("Predict if a customer is likely to churn based on their details.")

# ---------------------------
# User Input
# ---------------------------
credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650)
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 18, 92, 35)
tenure = st.slider("Tenure (years with bank)", 0, 10, 5)
balance = st.number_input("Balance", min_value=0.0, value=10000.0)
num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)

# ---------------------------
# Encode categorical variables
# ---------------------------
# Gender: Male=1, Female=0
gender_map = {"Male": 1, "Female": 0}

# Geography: One-hot encoding (same as training)
geography_map = {
    "France": [1, 0, 0],
    "Germany": [0, 1, 0],
    "Spain": [0, 0, 1]
}

# ---------------------------
# Create input array (9 features)
# ---------------------------
# Since RowNumber, CustomerId, Surname, Exited were removed, model expects:
# [CreditScore, Gender, Age, Tenure, Balance, NumProducts, HasCrCard, IsActiveMember, EstimatedSalary, Geography_onehot (3)] 
# Total features in scaler = 12 (if you included 3 geography columns)

input_data = [
    credit_score,
    gender_map[gender],
    age,
    tenure,
    balance,
    num_products,
    has_cr_card,
    is_active_member,
    estimated_salary,
    *geography_map[geography]  # 3 columns
]

# ---------------------------
# Scale input
# ---------------------------
input_scaled = scaler.transform([input_data])

# ---------------------------
# Prediction
# ---------------------------
if st.button("Predict Churn"):
    prediction = model.predict(input_scaled)[0][0]
    if prediction > 0.5:
        st.error(f"⚠️ This customer is likely to churn. (Probability: {prediction:.2f})")
    else:
        st.success(f"✅ This customer is not likely to churn. (Probability: {prediction:.2f})")
