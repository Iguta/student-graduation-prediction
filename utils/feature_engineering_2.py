### Load dataset, necessary libraries, and preprocessing to clean feature names

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df_student = pd.read_csv("/content/data.csv")

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
df_student.rename(columns=COLUMN_RENAMING_DICT, inplace=True)

# Remove "Enrolled" instances and map 'Target' variable to numeric values
df_student = df_student[df_student['target'].isin(['Dropout', 'Graduate'])]
df_student['target'] = df_student['target'].map({'Dropout': 0, 'Graduate': 1})

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
df_student['Course_Category'] = df_student['course'].map(course_numeric_assignment)

# Step 2: Create binary dummy variables for each major category, excluding Journalism (5)
# This will result in columns like 'agriculture_major', 'tech_major', etc.
df_student['agriculture_major'] = df_student['Course_Category'].apply(lambda x: 1 if x == 1 else 0)
df_student['tech_major'] = df_student['Course_Category'].apply(lambda x: 1 if x == 2 else 0)
df_student['health_major'] = df_student['Course_Category'].apply(lambda x: 1 if x == 3 else 0)
df_student['business_major'] = df_student['Course_Category'].apply(lambda x: 1 if x == 4 else 0)
df_student['education_major'] = df_student['Course_Category'].apply(lambda x: 1 if x == 6 else 0)

### consolidation of curricular unit variables

## creating 'unapproved_category' which classifies students into either 1 or zero unapproved courses or more (.75 correlation with target)

# Step 1: Create new columns by subtracting approved units from enrolled units for each semester
df_student['1st_sem_units_diff'] = df_student['1st_sem_units_enrolled'] - df_student['1st_sem_units_approved']
df_student['2nd_sem_units_diff'] = df_student['2nd_sem_units_enrolled'] - df_student['2nd_sem_units_approved']

# Step 2: Define the conditions for groups that make up new binary variable
# 1. Instances with an avg of one or fewer unapproved courses
one_or_fewer_unapproved = ((df_student['1st_sem_units_diff'] + df_student['2nd_sem_units_diff']) / 2 <= 1).astype(int)

# 2. Cases where both semesters enrolled units are zero (students who didn't enroll in any class shouldn't be in the same group as students who had no denied courses)
both_zero = ((df_student['1st_sem_units_enrolled'] == 0) & (df_student['2nd_sem_units_enrolled'] == 0)).astype(int)

# Step 3: Create a new column categorizing each instance into one of two groups
# 0 - More than one unapproved course OR both enrolled and approved are zero
# 1 - One or fewer unapproved courses
df_student['unapproved_category'] = 0
df_student.loc[(one_or_fewer_unapproved == 1) & (both_zero == 0), 'unapproved_category'] = 1

## creating 'academic_standing' variable based on grades from each semester

# Define the binning function for academic_standing
def assign_academic_standing(grade):
    if grade == 0:
        return 0  # No grades provided
    elif 0 < grade <= 11.5:
        return 1  # Bad
    elif 11.5 < grade <= 13.5:
        return 2  # Average
    else:
        return 3  # Good

# Apply the binning function to both semester grade columns and take the maximum
# to assign the best academic standing for each student across both semesters
df_student['academic_standing'] = df_student[['1st_sem_units_grade', '2nd_sem_units_grade']].applymap(assign_academic_standing).max(axis=1)