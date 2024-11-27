import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from utils.visualization import plot_category_proportion

def create_categorical_analysis_plots(df, preprocessor):
    st.header("Categorical Analysis")
    
    feature_groups = preprocessor.get_feature_groups()
    
    for group_name, features in feature_groups.items():
        st.subheader(f"{group_name}")
        
        categorical_features = [f for f in features 
                              if f in df.columns and 
                              (df[f].dtype == 'object' or df[f].nunique() < 10)]
        
        if categorical_features:
            selected_feature = st.selectbox(
                f"Select {group_name} Feature",
                categorical_features,
                key=f"select_{group_name}"
            )
            
            # Create proportion plot
            fig = plot_category_proportion(df, selected_feature, 'Target')
            st.plotly_chart(fig, use_container_width=True)
            
            # Add additional statistics
            st.subheader(f"Distribution of {selected_feature}")
            
            # Show value counts
            value_counts = df[selected_feature].value_counts()
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Count by Category")
                st.dataframe(value_counts)
            
            with col2:
                st.write("Percentage by Category")
                st.dataframe((value_counts / len(df) * 100).round(2))