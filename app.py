import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from utils import preprocessing, mappings

# Load the model and scaler
with open('logistic_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)
with open('scaler.pkl', 'rb') as scaler_file:
    loaded_scaler = pickle.load(scaler_file)

# Load the dataset 
@st.cache_data
def load_data():
    # If your data is stored locally, you could do:
    return preprocessing.clean_data()

data = load_data()

# CSS Styling for a better look and feel
st.markdown(
    """
    <style>
        .main {background-color: #f5f5f5;}
        .title {color: #2C3E50; font-size: 2.5em; font-weight: bold;}
        .subtitle {color: #34495E; font-size: 1.5em; font-weight: 600;}
        .prediction {color: #27AE60; font-size: 1.3em; font-weight: bold;}
    </style>
    """, unsafe_allow_html=True
)

# Function to show the prediction page
def show_prediction_page():
    st.markdown('<p class="title">🎓 Student Dropout Prediction App 🎓</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Enter the student’s details below to predict if they will graduate or drop out.</p>', unsafe_allow_html=True)

    # Organize inputs into columns
    col1, col2 = st.columns(2)
    with col1:
        units_approved_2nd = st.number_input("2nd Sem Units Approved", min_value=0.0, max_value=100.0, value=50.0)
        units_grade_2nd = st.number_input("2nd Sem Grade", min_value=0.0, max_value=20.0, value=10.0)
        units_approved_1st = st.number_input("1st Sem Units Approved", min_value=0.0, max_value=100.0, value=50.0)
        units_grade_1st = st.number_input("1st Sem Grade", min_value=0.0, max_value=20.0, value=10.0)
        course = st.number_input("Course", min_value=0, max_value=10, value=1)

    with col2:
        tuition_fees_up_to_date = st.selectbox("Tuition Fees Up to Date", ["No (0)", "Yes (1)"], index=1)
        scholarship_holder = st.selectbox("Scholarship Holder", ["No (0)", "Yes (1)"], index=0)
        enrollment_age = st.number_input("Enrollment Age", min_value=18, max_value=100, value=25)
        gender = st.selectbox("Gender", ["Female (0)", "Male (1)"], index=1)
        marital_status = st.selectbox("Marital Status", ["Single (0)", "Married (1)"], index=0)

    # Convert categorical options to numerical values
    tuition_fees_up_to_date = 1 if tuition_fees_up_to_date == "Yes (1)" else 0
    scholarship_holder = 1 if scholarship_holder == "Yes (1)" else 0
    gender = 1 if gender == "Male (1)" else 0
    marital_status = 1 if marital_status == "Married (1)" else 0

    # Collect input features in an array
    user_features = np.array([[units_approved_2nd, units_grade_2nd, units_approved_1st, units_grade_1st, 
                               course, tuition_fees_up_to_date, scholarship_holder, 
                               enrollment_age, gender, marital_status]])

    # Prediction button
    if st.button("Predict Graduation Status"):
        # Scale the input features
        scaled_features = loaded_scaler.transform(user_features)
        # Make prediction
        prediction = loaded_model.predict(scaled_features)[0]  # 0 = Dropout, 1 = Graduate
        prediction_proba = loaded_model.predict_proba(scaled_features)[0]

        st.markdown("---")  # Divider line
        st.markdown('<p class="subtitle">Prediction Results</p>', unsafe_allow_html=True)
        if prediction == 1:
            st.markdown(f'<p class="prediction">🎉 The student is predicted to **Graduate**! 🎓</p>', unsafe_allow_html=True)
        else:
            st.markdown(f'<p class="prediction">⚠️ The student is predicted to **Dropout**. ⚠️</p>', unsafe_allow_html=True)

        # Display probabilities
        st.markdown("### Prediction Probability")
        st.progress(prediction_proba[0])
        st.write(f"**Probability of Dropout:** {prediction_proba[0]:.2%}")
        st.progress(prediction_proba[1])
        st.write(f"**Probability of Graduation:** {prediction_proba[1]:.2%}")

# Function to show the analysis page
def show_analysis_page():
    st.markdown('<p class="title">📊 Student Data Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Visual analysis of predictor variables against the target variable (Graduate/Dropout).</p>', unsafe_allow_html=True)

    # Plot predictor variables vs. target variable
    predictors = [
        ('1st_sem_units_grade', "1st Sem Grade"),
        ('2nd_sem_units_grade', "2nd Sem Grade"),
        ('1st_sem_units_approved', "1st Sem Units Approved"),
        ('2nd_sem_units_approved', "2nd Sem Units Approved"),
        ('tuition_fees_up_to_date', "Tuition Fees Up-to-date"),
        ('scholarship_holder', "Scholarship Holder"),
        ('enrollment_age', "Enrollment Age"),
        ('gender', "Gender"),
        ('marital_status', "Marital Status"),
        ('course', "Course")
    ]


    # Calculate dropout rate for each course
    dropout_rate_by_course = (
        data[data['target'] == 0]
        .groupby('course')
        .size() / data.groupby('course').size() * 100
    ).reset_index(name='dropout_rate')
    # Get the course mapping from utils
    course_mapping = mappings.get_course_mapping()
    dropout_rate_by_course['course'] = dropout_rate_by_course['course'].map(course_mapping)

    # Plot the dropout rate by course
    st.subheader("Dropout Rate by Course")
    fig, ax = plt.subplots()
    sns.barplot(data=dropout_rate_by_course, x='course', y='dropout_rate', ax=ax, palette="viridis")
    ax.set_xlabel("Course")
    ax.set_ylabel("Dropout Rate (%)")
    ax.set_title("Dropout Rate by Course")
    # Rotate the y-axis labels for readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    st.pyplot(fig)

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Go to", ["Prediction", "Data Analysis"])

# Display the selected page
if page == "Prediction":
    show_prediction_page()
else:
    show_analysis_page()