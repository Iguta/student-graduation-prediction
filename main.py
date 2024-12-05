import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from components.analysis import create_categorical_analysis_plots
from utils.data_loader import load_data
from models.preprocessor import DataPreprocessor


def main():
    st.set_page_config(page_title="Student Dropout Prediction", layout="wide")
    
    st.title("🎓 Student Dropout Prediction System")
    st.markdown("""
    This application analyzes and predicts student dropout patterns using various academic and demographic factors.
    """)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", 
                           ["Data Overview",
                            "Categorical Analysis",
                            "Random Forest Model",
                            "Logistic Regression Model",
                            "Make Prediction"])
    
    try:
        # Load and preprocess data
        df = load_data("C:/Intro_DataScience/Week4/Academic_Success_Data.csv")
        df = df[df['Target'].isin(['Dropout', 'Graduate'])]
        df_processed = preprocessor.apply_mappings(df)
        
        if page == "Data Overview":
            st.header("Dataset Overview")
            
            # Basic statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Students", len(df))
            with col2:
                st.metric("Dropout Rate", f"{(df['Target'] == 'Dropout').mean():.1%}")
            with col3:
                st.metric("Graduate Rate", f"{(df['Target'] == 'Graduate').mean():.1%}")
            
            # Create two columns for the pie chart and sample data
            col_left, col_right = st.columns([1, 2])
            
            with col_left:
                # Data Distribution Pie Chart
                st.subheader("Data Distribution")
                target_counts = df['Target'].value_counts()
                fig = px.pie(
                    values=target_counts.values,
                    names=target_counts.index,
                    title="Distribution of Dropout vs Graduate",
                    color_discrete_map={'Dropout': '#FF6B6B', 'Graduate': '#4CAF50'}
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(
                    showlegend=False,
                    margin=dict(t=40, b=0, l=0, r=0)
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col_right:
                # Show sample data with categorical labels
                st.subheader("Sample Data")
                df_display = format_sample_data(df_processed)
                st.dataframe(df_display.head())
        
        elif page == "Categorical Analysis":
            create_categorical_analysis_plots(df_processed, preprocessor)
        
        elif page == "Random Forest Model":
            st.header("Random Forest Model")
            train_model(df_processed)
        
        elif page == "Logistic Regression Model":
            train_logistic_model(df_processed)
        
        elif page == "Make Prediction":
            st.header("Make Predictions")
            
            # Model selection
            model_type = st.radio(
                "Select Model for Prediction",
                ["Random Forest", "Logistic Regression"],
                horizontal=True
            )
            
            # Check for model files
            rf_model_exists = os.path.exists('trained_model.joblib')
            log_model_exists = os.path.exists('trained_logistic_model.joblib')
            
            if model_type == "Random Forest" and not rf_model_exists:
                st.warning("No trained Random Forest model found. Please train the model first.")
                if st.button("Go to Random Forest Training"):
                    st.session_state.page = "Random Forest Model"
                return
                
            if model_type == "Logistic Regression" and not log_model_exists:
                st.warning("No trained Logistic Regression model found. Please train the model first.")
                if st.button("Go to Logistic Regression Training"):
                    st.session_state.page = "Logistic Regression Model"
                return
            
            # Load appropriate model and make prediction
            if model_type == "Random Forest":
                make_prediction(df_processed, model_type='rf')
            else:
                make_prediction(df_processed, model_type='logistic')
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.info("Please check the data file path and format.")

if __name__ == "__main__":
    main()