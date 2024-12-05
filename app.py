import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
import os

from utils.preprocessing import load_data, engineer_features, prepare_features
from utils.mappings import (
    APPLICATION_MODE_MAPPING, 
    COURSE_MAPPING, 
    COURSE_CATEGORY_MAPPING,
    COURSE_CONDENSED_MAPPING,
    FEATURE_GROUPS
)
from utils.models.logistic_regression import *
from utils.visualizations import (
    plot_category_proportion,
    plot_feature_importance,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_prediction_probabilities,
    plot_feature_impact
)
from utils.pages import (
    display_data_overview,
    display_categorical_analysis,
    train_logistic_model,
    make_prediction
)

def main():
    st.set_page_config(page_title="Student Dropout Prediction", layout="wide")
    
    st.title("🎓 Student Dropout Prediction System")
    
    # Initialize navigation
    page = st.sidebar.radio("Go to", 
                          ["Data Overview",
                           "Data Analysis",
                           "Make Prediction"])
    
    try:
        # Load data
        df = load_data()
        df = df[df['Target'].isin(['Dropout', 'Graduate'])]
        
        if page == "Data Overview":
            display_data_overview(df)
        
        elif page == "Data Analysis":
            display_categorical_analysis(df)
        
        elif page == "Make Prediction":
            make_prediction(df)
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.info("Please check the data file path and format.")

# Add your display_data_overview, display_categorical_analysis, 
# train_logistic_model, and make_prediction functions here

if __name__ == "__main__":
    main()