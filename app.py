import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from utils import preprocessing, mappings
from utils import feature_engineering_2
from utils.feature_engineering_2 import plot_scatter_with_grade_avg
from streamlit_plotly_events import plotly_events

from utils.mappings import course_mapping
# Get the course mapping
course_mapping = course_mapping()

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

# # Function to show the prediction page
# def show_prediction_page():
#     st.markdown('<p class="title">🎓 Student Dropout Prediction App 🎓</p>', unsafe_allow_html=True)
#     st.markdown('<p class="subtitle">Enter the student’s details below to predict if they will graduate or drop out.</p>', unsafe_allow_html=True)

#     # Organize inputs into columns
#     col1, col2 = st.columns(2)
#     with col1:
#         units_approved_2nd = st.number_input("2nd Sem Units Approved", min_value=0.0, max_value=100.0, value=50.0)
#         units_grade_2nd = st.number_input("2nd Sem Grade", min_value=0.0, max_value=20.0, value=10.0)
#         units_approved_1st = st.number_input("1st Sem Units Approved", min_value=0.0, max_value=100.0, value=50.0)
#         units_grade_1st = st.number_input("1st Sem Grade", min_value=0.0, max_value=20.0, value=10.0)
#         course = st.selectbox(
#             "Course",
#             options=list(course_mapping.keys()),  # Use course codes as options
#             format_func=lambda x: course_mapping[x] if x in course_mapping else "Unknown",  # Display course names
#             index=0  # Default to the first option
#         )
#         units_enrolled_1st = st.number_input("Units Enrolled 1st Sem", min_value=0.0, max_value=100.0, value=50.0)
        

#     with col2:
#         tuition_fees_up_to_date = st.selectbox("Tuition Fees Up to Date", ["No (0)", "Yes (1)"], index=1)
#         scholarship_holder = st.selectbox("Scholarship Holder", ["No (0)", "Yes (1)"], index=0)
#         enrollment_age = st.number_input("Enrollment Age", min_value=18, max_value=100, value=25)
#         gender = st.selectbox("Gender", ["Female (0)", "Male (1)"], index=1)
#         marital_status = st.selectbox("Marital Status", ["Single (0)", "Married (1)"], index=0)
#         units_enrolled_2nd = st.number_input("Units Enrolled 2nd Sem", min_value=0.0, max_value=100.0, value=50.0)

#     # Convert categorical options to numerical values
#     tuition_fees_up_to_date = 1 if tuition_fees_up_to_date == "Yes (1)" else 0
#     scholarship_holder = 1 if scholarship_holder == "Yes (1)" else 0
#     gender = 1 if gender == "Male (1)" else 0
#     marital_status = 1 if marital_status == "Married (1)" else 0

#     # Compute new features
#     # Unapproved Category
#     units_diff_1st = units_enrolled_1st - units_approved_1st
#     units_diff_2nd = units_enrolled_2nd - units_approved_2nd
#     one_or_fewer_unapproved = ((units_diff_1st + units_diff_2nd) / 2 <= 1)
#     both_zero = (units_enrolled_1st == 0 and units_enrolled_2nd == 0)
#     unapproved_category = 1 if (one_or_fewer_unapproved and not both_zero) else 0

#     # Academic Standing
#     def assign_academic_standing(grade):
#         if grade == 0:
#             return 0  # No grades provided
#         elif 0 < grade <= 11.5:
#             return 1  # Bad
#         elif 11.5 < grade <= 13.5:
#             return 2  # Average
#         else:
#             return 3  # Good

#     academic_standing = max(assign_academic_standing(units_grade_1st), assign_academic_standing(units_grade_2nd))

#     # Collect input features in an array
#     user_features = np.array([[
#         unapproved_category, 
#         course,
#         tuition_fees_up_to_date, 
#         scholarship_holder,
#         enrollment_age,
#         gender,
#         marital_status,
#         academic_standing
#     ]])


#     # Prediction button
#     if st.button("Predict Graduation Status"):
#         # Scale the input features
#         scaled_features = loaded_scaler.transform(user_features)
#         # Make prediction
#         prediction = loaded_model.predict(scaled_features)[0]  # 0 = Dropout, 1 = Graduate
#         prediction_proba = loaded_model.predict_proba(scaled_features)[0]

#         st.markdown("---")  # Divider line
#         st.markdown('<p class="subtitle">Prediction Results</p>', unsafe_allow_html=True)
#         if prediction == 1:
#             st.markdown(f'<p class="prediction">🎉 The student is predicted to **Graduate**! 🎓</p>', unsafe_allow_html=True)
#         else:
#             st.markdown(f'<p class="prediction">⚠️ The student is predicted to **Dropout**. ⚠️</p>', unsafe_allow_html=True)

#         # Display probabilities
#         st.markdown("### Prediction Probability")
#         st.progress(prediction_proba[0])
#         st.write(f"**Probability of Dropout:** {prediction_proba[0]:.2%}")
#         st.progress(prediction_proba[1])
#         st.write(f"**Probability of Graduation:** {prediction_proba[1]:.2%}")

# Function to show the analysis page
# def show_analysis_page():
#     st.markdown('<p class="title">📊 Student Data Analysis</p>', unsafe_allow_html=True)
#     st.markdown('<p class="subtitle">Visual analysis of predictor variables against the target variable (Graduate/Dropout).</p>', unsafe_allow_html=True)

#     # Plot predictor variables vs. target variable
#     predictors = [
#         ('1st_sem_units_grade', "1st Sem Grade"),
#         ('2nd_sem_units_grade', "2nd Sem Grade"),
#         ('1st_sem_units_approved', "1st Sem Units Approved"),
#         ('2nd_sem_units_approved', "2nd Sem Units Approved"),
#         ('tuition_fees_up_to_date', "Tuition Fees Up-to-date"),
#         ('scholarship_holder', "Scholarship Holder"),
#         ('enrollment_age', "Enrollment Age"),
#         ('gender', "Gender"),
#         ('marital_status', "Marital Status"),
#         ('course', "Course")
#     ]


#     # Calculate dropout rate for each course
#     dropout_rate_by_course = (
#         data[data['target'] == 0]
#         .groupby('course')
#         .size() / data.groupby('course').size() * 100
#     ).reset_index(name='dropout_rate')
#     # Get the course mapping from utils
#     course_mapping = mappings.get_course_mapping()
#     dropout_rate_by_course['course'] = dropout_rate_by_course['course'].map(course_mapping)

#     # Plot the dropout rate by course
#     st.subheader("Dropout Rate by Course")
#     fig, ax = plt.subplots()
#     sns.barplot(data=dropout_rate_by_course, x='course', y='dropout_rate', ax=ax, palette="viridis")
#     ax.set_xlabel("Course")
#     ax.set_ylabel("Dropout Rate (%)")
#     ax.set_title("Dropout Rate by Course")
#     # Rotate the y-axis labels for readability
#     ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
#     st.pyplot(fig)


# Function to show Carson's Student Segmentation Analysis page
def show_prediction2_page():
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
    fig = feature_engineering_2.plot_category_proportion(df, category_col, target_col, mapping_dict)
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
        fig_scatter = plot_scatter_with_grade_avg(
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

# def display_data_overview():
#     st.header("Dataset Overview")
#     df = preprocessing.clean_data()

#     target_mapping = {0: "Dropout", 1: "Graduate"}
#     df['target'] = df['target'].map(target_mapping)

#     col1, col2, col3 = st.columns(3)
#     with col1:
#         st.metric("Total Students", len(df))
#     with col2:
#         st.metric("Dropout Rate", f"{(df['target'] == 'Dropout').mean():.1%}")
#     with col3:
#         st.metric("Graduate Rate", f"{(df['target'] == 'Graduate').mean():.1%}")
    
#     col_left, col_right = st.columns([1, 2])
#     with col_left:
#         target_counts = df['target'].value_counts()
#         fig = px.pie(values=target_counts.values, 
#                     names=target_counts.index,
#                     title="Dropout vs Graduate",
#                     color_discrete_map={'Dropout': '#FF6B6B', 'Graduate': '#4CAF50'})
#         st.plotly_chart(fig, use_container_width=True)
    
#     with col_right:
#         st.subheader("Sample Data")
#         df_display = format_sample_data(df)
#         st.dataframe(df.head())

#     # Load data
    
#     df = df[df['target'].isin(['Dropout', 'Graduate'])]

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Go to", ["Data Overview", "Segmentation Analysis","Prediction"])

# Display the selected page
if page == "Prediction":
    show_prediction_page()
elif page == "Segmentation Analysis":
    show_prediction2_page()
else:
    display_data_overview()

