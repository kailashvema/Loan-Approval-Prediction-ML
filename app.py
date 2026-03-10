import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("loan_data.csv")

# Encode categorical columns
encoder = LabelEncoder()
categorical_columns = [
    "Gender",
    "Married",
    "Education",
    "Self_Employed",
    "Property_Area",
    "Loan_Status"
]

for col in categorical_columns:
    df[col] = encoder.fit_transform(df[col])

# Features and target
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

# Train model
model = RandomForestClassifier()
model.fit(X, y)

st.title("Loan Approval Prediction App")

st.write("Enter applicant details to predict loan approval.")

# User Inputs
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.number_input("Applicant Income")
coapplicant_income = st.number_input("Coapplicant Income")
loan_amount = st.number_input("Loan Amount")
loan_term = st.selectbox("Loan Term", [360,180,240,120])
credit_history = st.selectbox("Credit History", [1,0])
property_area = st.selectbox("Property Area", ["Urban","Semiurban","Rural"])

# Convert inputs
input_data = pd.DataFrame({
    "Gender":[gender],
    "Married":[married],
    "Education":[education],
    "Self_Employed":[self_employed],
    "ApplicantIncome":[applicant_income],
    "CoapplicantIncome":[coapplicant_income],
    "LoanAmount":[loan_amount],
    "Loan_Amount_Term":[loan_term],
    "Credit_History":[credit_history],
    "Property_Area":[property_area]
})

for col in input_data.columns:
    if input_data[col].dtype == "object":
        input_data[col] = encoder.fit_transform(input_data[col])

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("Loan Approved")
    else:
        st.error("Loan Not Approved")