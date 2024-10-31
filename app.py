import pickle
import streamlit as st
import numpy as np
import pickle

# Load the model and scaler
with open('logistic_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)
with open('scaler.pkl', 'rb') as scaler_file:
    loaded_scaler = pickle.load(scaler_file)

st.title("Student Dropout Prediction App")

# User inputs for all 10 features
units_approved_2nd = st.number_input("2nd Sem Units Approved", min_value=0.0, max_value=100.0, value=50.0)
units_grade_2nd = st.number_input("2nd Sem Grade", min_value=0.0, max_value=20.0, value=10.0)
units_approved_1st = st.number_input("1st Sem Units Approved", min_value=0.0, max_value=100.0, value=50.0)
units_grade_1st = st.number_input("1st Sem Grade", min_value=0.0, max_value=20.0, value=10.0)
course = st.number_input("Course", min_value=0, max_value=10, value=1)  # adjust value range based on actual data
tuition_fees_up_to_date = st.selectbox("Tuition Fees Up to Date", [0, 1])
scholarship_holder = st.selectbox("Scholarship Holder", [0, 1])
enrollment_age = st.number_input("Enrollment Age", min_value=18, max_value=100, value=25)
gender = st.selectbox("Gender (0 for Female, 1 for Male)", [0, 1])
marital_status = st.selectbox("Marital Status", [0, 1])  # assuming binary for simplicity

# Collect all input features in an array
user_features = np.array([[units_approved_2nd, units_grade_2nd, units_approved_1st, units_grade_1st, 
                           course, tuition_fees_up_to_date, scholarship_holder, 
                           enrollment_age, gender, marital_status]])

# Scale the input features using the loaded scaler
scaled_features = loaded_scaler.transform(user_features)

# Make prediction with the loaded model
prediction = loaded_model.predict(scaled_features)[0]  # 0 = Dropout, 1 = Graduate
prediction_proba = loaded_model.predict_proba(scaled_features)[0]

# Display the prediction
st.subheader("Prediction")
st.write("The student is predicted to **Graduate**." if prediction == 1 else "The student is predicted to **Dropout**.")

# Display probabilities
st.subheader("Prediction Probability")
st.write(f"Probability of Dropout: {prediction_proba[0]:.2f}")
st.write(f"Probability of Graduation: {prediction_proba[1]:.2f}")