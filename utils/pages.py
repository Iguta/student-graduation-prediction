import streamlit as st
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import plotly.express as px
import pickle

from .preprocessing import engineer_features, prepare_features 
from .mappings import *
from utils import preprocessing, feature_engineering_2 as fe2, mappings
from streamlit_plotly_events import plotly_events

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
from .models.xgb_model import *

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

def get_course_code(course_name: str) -> int:
    """
    Map course name to its code.
    """
    name_to_code = {v: k for k, v in COURSE_MAPPING.items()}
    return name_to_code.get(course_name, None)

def make_prediction(df):
    st.markdown('<p class="title">🎓 Student Dropout Prediction App 🎓</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Enter student details to predict graduation status.</p>', unsafe_allow_html=True)

    # if not os.path.exists('xgb_model.pkl'):
    #     st.warning("No trained model found. Please train the model first.")
    #     return
    
    with open('xgb_model.pkl', 'rb') as file:
        model = pickle.load(file)
    print(f"Model loaded from 'xgb_model.pkl'")
    
    col1, col2 = st.columns(2)
    with col1:
        units_approved_2nd = st.number_input("2nd Sem Units Approved", min_value=0.0, max_value=100.0, value=50.0)
        units_grade_2nd = st.number_input("2nd Sem Grade", min_value=0.0, max_value=20.0, value=10.0)
        units_approved_1st = st.number_input("1st Sem Units Approved", min_value=0.0, max_value=100.0, value=50.0)
        units_grade_1st = st.number_input("1st Sem Grade", min_value=0.0, max_value=20.0, value=10.0)
        course = st.selectbox("Select a Course", options=list(COURSE_MAPPING.values()))

    with col2:
        tuition_fees_up_to_date = st.selectbox("Tuition Fees Up to Date", ["No", "Yes"], index=1)
        scholarship_holder = st.selectbox("Scholarship Holder", ["No", "Yes"], index=0)
        enrollment_age = st.number_input("Enrollment Age", min_value=18, max_value=100, value=25)
        gender = st.selectbox("Gender", ["Female", "Male"], index=1)
        marital_status = st.selectbox("Marital Status", ["Single", "Married"], index=0)

    course_code = get_course_code(course)
    input_data = pd.DataFrame({
        '2nd_sem_units_approved': [units_approved_2nd],
        '2nd_sem_units_grade': [units_grade_2nd],
        '1st_sem_units_approved': [units_approved_1st],
        '1st_sem_units_grade': [units_grade_1st],
        'course': [course_code],
        'tuition_fees_up_to_date': [1 if tuition_fees_up_to_date == "Yes" else 0],
        'scholarship_holder': [1 if scholarship_holder == "Yes" else 0],
        'enrollment_age': [enrollment_age],
        'gender': [1 if gender == "Male" else 0],
        'marital_status': [1 if marital_status == "Married" else 0]
    })

    if st.button("Predict Graduation Status"):
        prediction, probabilities = predict(model, input_data)
        print(probabilities)
        
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

# Function to show Carson's Student Segmentation Analysis page
def Segment_Analysis():
    st.markdown('<p class="title"> 📊 Student Segmentation 📊 </p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Select a category.</p>', unsafe_allow_html=True)

    # dropdown for selecting a category
    category_options = mappings.category_dict()
    selected_option = st.selectbox("Categorical Selection:",
                                     list(mappings.category_dict().values())
    )

    # Map the selected option to the corresponding column and mapping dictionary
    for key, value in category_options.items():
        if value == selected_option:
            category_col = "course_category" if key == "course_condensed_mapping" else key.replace("_mapping", "")  # Extract the column name
            mapping_dict = getattr(mappings, key)()  # Dynamically call the corresponding mapping function
            break

    # predetermined argument
    df = preprocessing.clean_data()
    target_col = "target"

    # Calculate the number of categories in the first plot to determine height
    num_categories = len(df[category_col].unique())  # Or use any logic related to the first plot

    # Set the height dynamically based on the number of categories in the first plot
    calculated_height = max(860, num_categories * 35 + 150)  # Minimum height is 400, scale up based on categories

    # generate plot
    fig = fe2.plot_category_proportion(df, category_col, target_col, mapping_dict)
    selected_points = plotly_events(
        fig,
        click_event=True,  # Enables click event capture
        hover_event=False,  # Disable hover events (optional)
        select_event=False,  # Disable select events (optional)
        override_height=calculated_height,  # Adjust the height of the chart (optional)
        override_width="115%"  # Adjust the width of the chart (optional)
    )

    st.write("Click on a bar above to filter the scatter plot below.")

    # Check if any points are clicked
    if selected_points:
        selected_category = selected_points[0]["y"]  # Capture the clicked category
        st.write(f"You selected: {selected_category}")

        # Generate the second scatter plot dynamically
        fig_scatter = fe2.plot_scatter_with_grade_avg(
            df,
            category_col=category_col,
            filter_value=selected_category,
            x_col='prev_qualification_grade',
            y_col='adm_grade',
            target_col='target'
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.write("No category selected yet.")

def format_sample_data(df):
    """Format sample data by converting numerical categories to labels"""
    df_display = df.copy()
    
    binary_map = {0: 'No', 1: 'Yes'}
    gender_map = {0: 'Female', 1: 'Male'}
    
    binary_columns = [
        'Displaced', 'Educational special needs',
        'Debtor', 'Tuition fees up to date', 'Gender', 'Scholarship holder',
        'International'
    ]
    
    for col in binary_columns:
        if col in df_display.columns:
            df_display[col] = df_display[col].map(gender_map if col == 'Gender' else binary_map)
    
    display_columns = [
        'Course', 'Previous qualification', 'Gender', 
        'Age at enrollment', 'International', 'Scholarship holder', 
        'Tuition fees up to date', 'Displaced', 'Educational special needs', 
        'Debtor', 'Target'
    ]
    
    display_columns = [col for col in display_columns if col in df_display.columns]
    return df_display[display_columns]