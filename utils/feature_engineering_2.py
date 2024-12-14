### Load dataset, necessary libraries, and preprocessing to clean feature names

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def get_feature_groups():
    """Return groups of features for analysis"""
    return {
        'Demographic Features': [
            'marital_status',
            'Age at enrollment',
            'International',
            'Displaced'
        ],
        'Application Features': [
            'Application mode',
            'Course_Category',
            'Course_Name',
            'Previous qualification (grade)',
            'Admission grade'
        ],
        'Academic Performance': [
            'Curricular units 1st sem (enrolled)',
            'Curricular units 1st sem (evaluations)',
            'Curricular units 1st sem (approved)',
            'Curricular units 1st sem (grade)',
            'Curricular units 2nd sem (enrolled)',
            'Curricular units 2nd sem (evaluations)',
            'Curricular units 2nd sem (approved)',
            'Curricular units 2nd sem (grade)',
            'first_sem_success_ratio',
            'second_sem_success_ratio',
            'average_grade',
            'performance_change'
        ],
        'Economic Factors': [
            'Scholarship holder',
            'Tuition fees up to date',
            'Debtor',
            'Unemployment rate',
            'Inflation rate',
            'GDP',
            'economic_factor'
        ]
    }

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
