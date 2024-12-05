import pandas as pd
import streamlit as st

class DataPreprocessor:
    def __init__(self):
        # Application mode mapping
        self.application_mode_mapping = {
            1: '1st phase - general',
            2: 'Ordinance 612/93',
            5: '1st phase - Azores',
            7: 'Other higher courses',
            10: 'Ordinance 854-B/99',
            15: 'Intl. student (bachelor)',
            16: '1st phase - Madeira',
            17: '2nd phase - general',
            18: '3rd phase - general',
            26: 'Ordinance 533-A/99 (Plan)',
            27: 'Ordinance 533-A/99 (Institution)',
            39: 'Over 23 years old',
            42: 'Transfer',
            43: 'Change of course',
            44: 'Technological diploma',
            51: 'Change institution/course',
            53: 'Short cycle diploma',
            57: 'Change institution (Intl.)'
        }
        
        # Marital status mapping
        self.marital_status_mapping = {
            1: 'Single',
            2: 'Married',
            3: 'Widower',
            4: 'Divorced',
            5: 'Facto union',
            6: 'Legally separated'
        }
        
        # Course mappings
        self.course_mapping = {
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
        
        self.course_numeric_assignment = {
            33: 1, 171: 2, 8014: 3, 9003: 1, 9070: 2, 9085: 3,
            9119: 2, 9130: 1, 9147: 4, 9238: 3, 9254: 4, 9500: 3,
            9556: 3, 9670: 4, 9773: 5, 9853: 6, 9991: 4
        }
        
        self.course_condensed_mapping = {
            1: 'Agriculture',
            2: 'Tech',
            3: 'Health',
            4: 'Business',
            5: 'Journalism',
            6: 'Education'
        }
    
    def apply_mappings(self, df):
        """Apply all mappings to the dataframe"""
        df = df.copy()
        
        # Apply basic mappings
        df['Application mode'] = df['Application mode'].map(self.application_mode_mapping)
        df['Marital status'] = df['Marital status'].map(self.marital_status_mapping)
        df['Course_Name'] = df['Course'].map(self.course_mapping)
        
        # Apply course categorization
        df['Course_Category'] = df['Course'].map(self.course_numeric_assignment)
        df['Course_Category'] = df['Course_Category'].map(self.course_condensed_mapping)
        
        # Fill missing values
        categorical_columns = [
            'Application mode', 'Marital status', 'Course_Name', 
            'Course_Category'
        ]
        
        for col in categorical_columns:
            df[col] = df[col].fillna('Unknown')
        
        return df
    
    def get_feature_groups(self):
        """Return groups of features for analysis"""
        return {
            'Demographic Features': [
                'Marital status',
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