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