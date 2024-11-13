def format_sample_data(df):
    """Format sample data by converting numerical categories to labels"""
    df_display = df.copy()
    
    # Binary columns
    binary_columns = [
        'Displaced', 'Educational special needs',
        'Debtor', 'Tuition fees up to date', 'Gender', 'Scholarship holder',
        'International'
    ]
    
    binary_map = {
        0: 'No',
        1: 'Yes'
    }
    
    gender_map = {
        0: 'Female',
        1: 'Male'
    }
    
    # Course mapping
    course_map = {
        33: 'Biofuel Production Tech',
        171: 'Animation & Multimedia',
        8014: 'Social Service (evening)',
        9003: 'Agronomy',
        9070: 'Comm. Design',
        9085: 'Veterinary Nursing',
        9119: 'Informatics Eng.',
        9130: 'Equinculture',
        9147: 'Mgmt',
        9238: 'Social Service',
        9254: 'Tourism',
        9500: 'Nursing',
        9556: 'Oral Hygiene',
        9670: 'Advertising & Marketing',
        9773: 'Journalism & Comm.',
        9853: 'Basic Education',
        9991: 'Mgmt (evening)'
    }
    
    # Previous qualification mapping
    prev_qual_map = {
        1: 'Secondary Education',
        2: "Bachelor's Degree",
        3: 'Degree',
        4: "Master's",
        5: 'Doctorate',
        6: 'Higher Education Frequency',
        9: '12th Year (Not Completed)',
        10: '11th Year (Not Completed)',
        12: 'Other - 11th Year',
        14: '10th Year',
        15: '10th Year (Not Completed)',
        19: 'Basic Education 3rd Cycle',
        38: 'Basic Education 2nd Cycle',
        39: 'Tech Specialization',
        40: 'Degree (1st Cycle)',
        42: 'Professional Tech',
        43: 'Master (2nd Cycle)'
    }
    
    try:
        # Convert numeric columns and apply mappings
        df_display['Course'] = pd.to_numeric(df_display['Course'], errors='coerce')
        df_display['Previous qualification'] = pd.to_numeric(df_display['Previous qualification'], errors='coerce')
        
        # Apply mappings
        df_display['Course'] = df_display['Course'].map(course_map)
        df_display['Previous qualification'] = df_display['Previous qualification'].map(prev_qual_map)
        
        # Apply binary mappings
        for col in binary_columns:
            if col in df_display.columns:
                df_display[col] = pd.to_numeric(df_display[col], errors='coerce')
                if col == 'Gender':
                    df_display[col] = df_display[col].map(gender_map)
                else:
                    df_display[col] = df_display[col].map(binary_map)
    
    except Exception as e:
        print(f"Error in mapping: {e}")
    
    # Select and reorder columns for display
    display_columns = [
        'Course', 
        'Previous qualification',
        'Gender', 
        'Age at enrollment', 
        'International',
        'Scholarship holder', 
        'Tuition fees up to date', 
        'Displaced',
        'Educational special needs', 
        'Debtor', 
        'Target'
    ]
    
    # Only include columns that exist in the dataframe
    display_columns = [col for col in display_columns if col in df_display.columns]
    df_display = df_display[display_columns]
    
    return df_display