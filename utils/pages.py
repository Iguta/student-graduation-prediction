import streamlit as st
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import plotly.express as px

from .preprocessing import engineer_features, prepare_features 

from .visualizations import (
    plot_category_proportion,
    plot_feature_importance,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_prediction_probabilities,
    plot_feature_impact
)
from .mappings import FEATURE_GROUPS
from .data_formatting import format_sample_data
from .models.logistic_regression import LogisticRegressionModel

def display_data_overview(df):
    st.header("Dataset Overview")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Students", len(df))
    with col2:
        st.metric("Dropout Rate", f"{(df['Target'] == 'Dropout').mean():.1%}")
    with col3:
        st.metric("Graduate Rate", f"{(df['Target'] == 'Graduate').mean():.1%}")
    
    col_left, col_right = st.columns([1, 2])
    with col_left:
        target_counts = df['Target'].value_counts()
        fig = px.pie(values=target_counts.values, 
                    names=target_counts.index,
                    title="Distribution of Dropout vs Graduate",
                    color_discrete_map={'Dropout': '#FF6B6B', 'Graduate': '#4CAF50'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col_right:
        st.subheader("Sample Data")
        df_display = format_sample_data(df)
        st.dataframe(df_display.head())

def display_categorical_analysis(df):
    st.header("Categorical Analysis")
    
    for group_name, features in FEATURE_GROUPS.items():
        st.subheader(f"{group_name}")
        categorical_features = [f for f in features 
                              if f in df.columns and 
                              (df[f].dtype == 'object' or df[f].nunique() < 10)]
        
        if categorical_features:
            selected_feature = st.selectbox(f"Select {group_name} Feature",
                                          categorical_features,
                                          key=f"select_{group_name}")
            fig = plot_category_proportion(df, selected_feature, 'Target')
            st.plotly_chart(fig, use_container_width=True)


def make_prediction(df):
    st.markdown('<p class="title">🎓 Student Dropout Prediction App 🎓</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Enter student details to predict graduation status.</p>', unsafe_allow_html=True)

    if not os.path.exists('model.pkl'):
        st.warning("No trained model found. Please train the model first.")
        return

    model = LogisticRegressionModel.load()
    
    col1, col2 = st.columns(2)
    with col1:
        units_approved_2nd = st.number_input("2nd Sem Units Approved", min_value=0.0, max_value=100.0, value=50.0)
        units_grade_2nd = st.number_input("2nd Sem Grade", min_value=0.0, max_value=20.0, value=10.0)
        units_approved_1st = st.number_input("1st Sem Units Approved", min_value=0.0, max_value=100.0, value=50.0)
        units_grade_1st = st.number_input("1st Sem Grade", min_value=0.0, max_value=20.0, value=10.0)
        course = st.number_input("Course", min_value=0, max_value=10, value=1)

    with col2:
        tuition_fees_up_to_date = st.selectbox("Tuition Fees Up to Date", ["No", "Yes"], index=1)
        scholarship_holder = st.selectbox("Scholarship Holder", ["No", "Yes"], index=0)
        enrollment_age = st.number_input("Enrollment Age", min_value=18, max_value=100, value=25)
        gender = st.selectbox("Gender", ["Female", "Male"], index=1)
        marital_status = st.selectbox("Marital Status", ["Single", "Married"], index=0)

    input_data = pd.DataFrame({
        '2nd_sem_units_approved': [units_approved_2nd],
        '2nd_sem_units_grade': [units_grade_2nd],
        '1st_sem_units_approved': [units_approved_1st],
        '1st_sem_units_grade': [units_grade_1st],
        'course': [course],
        'tuition_fees_up_to_date': [1 if tuition_fees_up_to_date == "Yes (1)" else 0],
        'scholarship_holder': [1 if scholarship_holder == "Yes (1)" else 0],
        'enrollment_age': [enrollment_age],
        'gender': [1 if gender == "Male (1)" else 0],
        'marital_status': [1 if marital_status == "Married (1)" else 0]
    })

    if st.button("Predict Graduation Status"):
        prediction, probabilities = model.predict(input_data)
        
        st.markdown("---")
        st.markdown('<p class="subtitle">Prediction Results</p>', unsafe_allow_html=True)
        
        if prediction[0] == 1:
            st.markdown('<p class="prediction">🎉 The student is predicted to **Graduate**! 🎓</p>', 
                       unsafe_allow_html=True)
        else:
            st.markdown('<p class="prediction">⚠️ The student is predicted to **Dropout**. ⚠️</p>', 
                       unsafe_allow_html=True)

        st.markdown("### Prediction Probability")
        st.progress(1 - probabilities[0])
        st.write(f"**Probability of Dropout:** {1 - probabilities[0]:.2%}")
        st.progress(probabilities[0])
        st.write(f"**Probability of Graduation:** {probabilities[0]:.2%}")