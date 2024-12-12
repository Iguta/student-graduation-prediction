### Load dataset, necessary libraries, and preprocessing to clean feature names

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# df_students = pd.read_csv("/content/data.csv")
from utils.preprocessing import *
# column renaming dictionary
COLUMN_RENAMING_DICT = {
    "Marital status": "marital_status",
    "Application mode": "app_mode",
    "Application order": "app_order",
    "Course": "course",
    "Daytime/evening attendance\t":"attendance",
    "Previous qualification":"prev_qualification",
    "Previous qualification (grade)":"prev_qualification_grade",
    "Nacionality":"nationality",
    "Mother's qualification":"mother_qualification",
    "Father's qualification":"father_qualification",
    "Mother's occupation":"mother_occupation",
    "Father's occupation":"father_occupation",
    "Admission grade":"adm_grade",
    "Displaced":"displaced",
    "Educational special needs":"educational_special_needs",
    "Debtor":"debtor",
    "Tuition fees up to date":"tuition_fees_up_to_date",
    "Gender":"gender",
    "Scholarship holder":"scholarship_holder",
    "Age at enrollment":"enrollment_age",
    "International":"international",
    "Curricular units 1st sem (credited)":"1st_sem_units_credited",
    "Curricular units 1st sem (enrolled)":"1st_sem_units_enrolled",
    "Curricular units 1st sem (evaluations)":"1st_sem_units_evaluations",
    "Curricular units 1st sem (approved)":"1st_sem_units_approved",
    "Curricular units 1st sem (grade)":"1st_sem_units_grade",
    "Curricular units 1st sem (without evaluations)":"1st_sem_units_no_evaluations",
    "Curricular units 2nd sem (credited)":"2nd_sem_units_credited",
    "Curricular units 2nd sem (enrolled)":"2nd_sem_units_enrolled",
    "Curricular units 2nd sem (evaluations)":"2nd_sem_units_evaluations",
    "Curricular units 2nd sem (approved)":"2nd_sem_units_approved",
    "Curricular units 2nd sem (grade)":"2nd_sem_units_grade",
    "Curricular units 2nd sem (without evaluations)":"2nd_sem_units_no_evaluations",
    "Unemployment rate":"unemployment_rate",
    "Inflation rate":"inflation_rate",
    "GDP":"gdp",
    "Target":"target"
}

# Renaming columns in the dataframe
df_students = load_data()
df_students.rename(columns=COLUMN_RENAMING_DICT, inplace=True)

# Remove "Enrolled" instances and map 'Target' variable to numeric values
df_students = df_students[df_students['target'].isin(['Dropout', 'Graduate'])]
df_students['target'] = df_students['target'].map({'Dropout': 0, 'Graduate': 1})

### remapping of course variable to consolidate classes and allow modelling with dummy variables

# Updated course mapping, excluding Journalism as the least frequent
course_numeric_assignment = {
    33: 1,   # Agriculture (Biofuel Production Technologies)
    171: 2,  # Tech (Animation and Multimedia Design)
    8014: 3, # Health (Social Service evening)
    9003: 1, # Agriculture (Agronomy)
    9070: 2, # Tech (Communication Design)
    9085: 3, # Health (Veterinary Nursing)
    9119: 2, # Tech (Informatics Engineering)
    9130: 1, # Agriculture (Equinculture)
    9147: 4, # Business (Management)
    9238: 3, # Health (Social Service)
    9254: 4, # Business (Tourism)
    9500: 3, # Health (Nursing)
    9556: 3, # Health (Oral Hygiene)
    9670: 4, # Business (Advertising and Marketing Management)
    9773: 5, # Journalism (excluded from dummy variables)
    9853: 6, # Education (Basic Education)
    9991: 4  # Business (Management evening)
}

# Step 1: Map each specific course code to one of the 6 categories (1-6)
df_students['Course_Category'] = df_students['course'].map(course_numeric_assignment)

# Step 2: Create binary dummy variables for each major category, excluding Journalism (5)
# This will result in columns like 'agriculture_major', 'tech_major', etc.
df_students['agriculture_major'] = df_students['Course_Category'].apply(lambda x: 1 if x == 1 else 0)
df_students['tech_major'] = df_students['Course_Category'].apply(lambda x: 1 if x == 2 else 0)
df_students['health_major'] = df_students['Course_Category'].apply(lambda x: 1 if x == 3 else 0)
df_students['business_major'] = df_students['Course_Category'].apply(lambda x: 1 if x == 4 else 0)
df_students['education_major'] = df_students['Course_Category'].apply(lambda x: 1 if x == 6 else 0)

### consolidation of curricular unit variables

## creating 'unapproved_category' which classifies students into either 1 or zero unapproved courses or more (.75 correlation with target)

# # Step 1: Create new columns by subtracting approved units from enrolled units for each semester
# df_students['1st_sem_units_diff'] = df_students['1st_sem_units_enrolled'] - df_students['1st_sem_units_approved']
# df_students['2nd_sem_units_diff'] = df_students['2nd_sem_units_enrolled'] - df_students['2nd_sem_units_approved']

# # Step 2: Define the conditions for groups that make up new binary variable
# # 1. Instances with an avg of one or fewer unapproved courses
# one_or_fewer_unapproved = ((df_students['1st_sem_units_diff'] + df_students['2nd_sem_units_diff']) / 2 <= 1).astype(int)

# # 2. Cases where both semesters enrolled units are zero (students who didn't enroll in any class shouldn't be in the same group as students who had no denied courses)
# both_zero = ((df_students['1st_sem_units_enrolled'] == 0) & (df_students['2nd_sem_units_enrolled'] == 0)).astype(int)

# # Step 3: Create a new column categorizing each instance into one of two groups
# # 0 - More than one unapproved course OR both enrolled and approved are zero
# # 1 - One or fewer unapproved courses
# df_students['unapproved_category'] = 0
# df_students.loc[(one_or_fewer_unapproved == 1) & (both_zero == 0), 'unapproved_category'] = 1

# ## creating 'academic_standing' variable based on grades from each semester

# # Define the binning function for academic_standing
# def assign_academic_standing(grade):
#     if grade == 0:
#         return 0  # No grades provided
#     elif 0 < grade <= 11.5:
#         return 1  # Bad
#     elif 11.5 < grade <= 13.5:
#         return 2  # Average
#     else:
#         return 3  # Good

# # Apply the binning function to both semester grade columns and take the maximum
# # to assign the best academic standing for each student across both semesters
# df_students['academic_standing'] = df_students[['1st_sem_units_grade', '2nd_sem_units_grade']].applymap(assign_academic_standing).max(axis=1)

############################################################################################

### Plot 1: Categorical proportions with distributions ###

import plotly.graph_objects as go
import pandas as pd
def plot_category_proportion(df, category_col, target_col, mapping_dict=None):
    """
    Create a horizontal stacked bar chart showing the proportion of target categories
    for each category, along with total counts.
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame containing the data
    category_col : str
        Name of the column containing categories
    target_col : str
        Name of the column containing target values (e.g., 'Graduate'/'Dropout')
    mapping_dict : dict, optional
        Dictionary to map category codes to their descriptions
    """
    if mapping_dict is not None and not isinstance(mapping_dict, dict):
        raise ValueError(f"Expected mapping_dict to be a dictionary, but got {type(mapping_dict)}")
    
    if category_col not in df.columns:
        raise ValueError(f"Column '{category_col}' not found in DataFrame")
    
    if target_col not in df.columns:
        raise ValueError(f"Column '{target_col}' not found in DataFrame")

    # If a mapping dictionary is provided, map the category column to the proper labels
    if mapping_dict:
        df[category_col] = df[category_col].map(mapping_dict)

    # Map the target column to labels for readability
    target_mapping = {0: "Dropout", 1: "Graduate"}
    df[target_col] = df[target_col].map(target_mapping)

    # Calculate counts for each category and target combination
    category_counts = df.groupby([category_col, target_col]).size().unstack(fill_value=0)
    # Sort the categories based on total counts, highest to lowest
    category_counts['Total'] = category_counts.sum(axis=1)
    category_counts = category_counts.sort_values(by='Total', ascending=True).drop(columns='Total')
    # Calculate percentages
    category_percentages = category_counts.div(category_counts.sum(axis=1), axis=0) * 100
    # Create the stacked bar chart
    fig = go.Figure()
    # Add dropout bars
    fig.add_trace(go.Bar(
        y=category_counts.index,
        x=category_percentages['Dropout'] if 'Dropout' in category_percentages.columns else [],
        name='Dropout',
        orientation='h',
        marker_color='salmon',
        text=category_percentages['Dropout'].round(1).astype(str) + '%' if 'Dropout' in category_percentages.columns else [],
        textposition='inside',
        insidetextanchor='middle'
    ))
    # Add graduate bars
    fig.add_trace(go.Bar(
        y=category_counts.index,
        x=category_percentages['Graduate'] if 'Graduate' in category_percentages.columns else [],
        name='Graduate',
        orientation='h',
        marker_color='lightblue',
        text=category_percentages['Graduate'].round(1).astype(str) + '%' if 'Graduate' in category_percentages.columns else [],
        textposition='inside',
        insidetextanchor='middle'
    ))
    # Add total counts as text on the right side
    total_counts = category_counts.sum(axis=1)
    # Add annotations for total counts
    annotations = [
        dict(
            x=100,
            y=i,
            text=str(count),
            xanchor='left',
            yanchor='middle',
            showarrow=False,
            font=dict(size=10),
            xshift=10
        )
        for i, count in enumerate(total_counts)
    ]
    # Add "Count:" label at the top
    annotations.append(
        dict(
            x=100,
            y=len(total_counts),
            text="Count:",
            xanchor='left',
            yanchor='middle',
            showarrow=False,
            font=dict(size=10),
            xshift=10,
            bordercolor='black',
            borderwidth=1,
            borderpad=4,
            bgcolor='white'
        )
    )
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'Proportion of {category_col} by {target_col}',
            font=dict(size=16)
        ),
        xaxis=dict(
            title='Proportion (%)',
            range=[0, 110],  # Extended range to accommodate count labels
            titlefont=dict(size=12)
        ),
        yaxis=dict(
            title='',
            titlefont=dict(size=18),
            tickfont=dict(size=10),
            tickangle = -40
        ),

        barmode='stack',
        showlegend=True,
        legend=dict(
            title=target_col,
            yanchor="bottom",
            y=0.5,
            xanchor="right",
            x=1.2
        ),
        annotations=annotations,
        height=400 + len(category_counts) * 20 + 100,  # Dynamically adjust height based on number of categories
        margin=dict(r=200)  # Add right margin for count labels
    )
    return fig

#################################################################################################

### Plot 2: Scatter representation that dynamically adjust to segment selection to see academic performance ###

from scipy.stats import linregress

def plot_scatter_with_grade_avg(df, category_col, filter_value, x_col, y_col, target_col):
    """
    Create a scatter plot for a specific class in a given category,
    differentiating dropouts and graduates by color and marker style, with enhanced hover labels including grade average.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame containing the data
    category_col : str
        Name of the column containing the category
    filter_value : str
        Specific value in the category to filter for
    x_col : str
        Name of the column for the x-axis
    y_col : str
        Name of the column for the y-axis
    target_col : str
        Name of the target column to distinguish classes (e.g., 'Dropout', 'Graduate')
    """

    # R² threshold for displaying trendlines
    r2_threshold = 0.4

    # Map gender and tuition fees to human-readable labels
    gender_mapping = {1: 'Male', 0: 'Female'}
    tuition_mapping = {1: 'Yes', 0: 'No'}
    df['gender'] = df['gender'].map(gender_mapping)
    df['tuition_fees_up_to_date'] = df['tuition_fees_up_to_date'].map(tuition_mapping)

    # Compute Grade avg this year
    df['Grade avg this year'] = (df['2nd_sem_units_grade'] + df['1st_sem_units_grade']) / 2

    # Filter the DataFrame for the specified class
    filtered_df = df[df[category_col] == filter_value]

    # Separate data into Dropouts and Graduates
    dropouts = filtered_df[filtered_df[target_col] == 'Dropout']
    graduates = filtered_df[filtered_df[target_col] == 'Graduate']

    # Construct hover text for Dropouts
    dropout_hover_text = [
        f"Previous qualification (grade): {row[x_col]}<br>"
        f"Admission grade: {row[y_col]}<br>"
        f"Gender: {row['gender']}<br>"
        f"Tuition fees up to date: {row['tuition_fees_up_to_date']}<br>"
        f"Age at enrollment: {row['enrollment_age']}<br>"
        f"Grade avg this year: {row['Grade avg this year']:.2f}"
        for _, row in dropouts.iterrows()
    ]

    # Construct hover text for Graduates
    graduate_hover_text = [
        f"Previous qualification (grade): {row[x_col]}<br>"
        f"Admission grade: {row[y_col]}<br>"
        f"Gender: {row['gender']}<br>"
        f"Tuition fees up to date: {row['tuition_fees_up_to_date']}<br>"
        f"Age at enrollment: {row['enrollment_age']}<br>"
        f"Grade avg this year: {row['Grade avg this year']:.2f}"
        for _, row in graduates.iterrows()
    ]

    # Create the scatter plot
    fig = go.Figure()

    # Add Dropouts
    fig.add_trace(go.Scatter(
        x=dropouts[x_col],
        y=dropouts[y_col],
        mode='markers',
        marker=dict(size=10, color='red', symbol='x'),
        name='Dropout',
        text=dropout_hover_text,
        hovertemplate="%{text}<extra></extra>"
    ))

    # Add Graduates
    fig.add_trace(go.Scatter(
        x=graduates[x_col],
        y=graduates[y_col],
        mode='markers',
        marker=dict(size=6, color='blue', symbol='circle'),
        name='Graduate',
        text=graduate_hover_text,
        hovertemplate="%{text}<extra></extra>"
    ))

    # Function to add a trendline if R^2 is above the threshold
    def add_trendline(data, color, name):
        if len(data) > 2:
            # Perform linear regression
            slope, intercept, r_value, _, _ = linregress(data[x_col], data[y_col])
            r_squared = r_value ** 2

            # Add trendline only if R^2 is above the threshold
            if r_squared >= r2_threshold:
                x_smooth = np.linspace(data[x_col].min(), data[x_col].max(), 300)
                y_smooth = slope * x_smooth + intercept

                fig.add_trace(go.Scatter(
                    x=x_smooth,
                    y=y_smooth,
                    mode='lines',
                    line=dict(color=color, width=3, dash='solid'),
                    name=f'{name} Trendline (R²={r_squared:.2f})'
                ))

    # Add trendlines for Dropouts and Graduates
    add_trendline(dropouts, 'salmon', 'Dropout')
    add_trendline(graduates, 'lightblue', 'Graduate')

    # Update the layout
    fig.update_layout(
        title=f"Scatter Plot for {filter_value} in {category_col}",
        xaxis=dict(title=x_col),
        yaxis=dict(title=y_col),
        legend=dict(title="Target"),
        height=600,
        width=800
    )

    return fig
