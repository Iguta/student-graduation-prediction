import pandas as pd
from pandas.api.types import is_numeric_dtype
import pickle

import os 
from utils.b2 import B2
from dotenv import load_dotenv

#load our environment variables
load_dotenv()

# column renaming dictionary
COLUMN_RENAMING_DICT = {
    "Marital status": "marital_status", 
    "Application mode": "app_mode",
    "Application order": "app_order",
    "Course": "course",
    "Daytime/evening attendance":"attendance",
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

#function to load our dataset
def load_data() -> pd.DataFrame:
    #reading the data set and returning a pandas data frame
    # try:
    #     df = pd.read_csv("././data/Student_Academic_Success.csv")
    #     return df
    # except FileNotFoundError:
    #     return pd.DataFrame()
    #     print("File not found")
    # except Exception as e:
    #     return pd.DataFrame()
    #     print("An unexpected error occured")

    #Intialize a B2 connection
    try:
        b2_instance = B2(endpoint=os.environ['B2_ENDPOINT'],
        key_id=os.environ['B2_KEYID'],
        secret_key=os.environ['B2_APPKEY'])
        #select the bucket
        b2_instance.set_bucket(os.environ['B2_BUCKET_NAME'])

        # List files in the bucket (optional, to confirm your file exists)
        print("Files in bucket:", b2_instance.list_files())
        df_student = b2_instance.get_df(os.environ['REMOTE_CSV_PATH'])

        print(df_student.head())
        return df_student
    except Exception as e:
        print(f"An error occured:{e}")


#function to rename columns
def rename_columns(df:pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=COLUMN_RENAMING_DICT)

#function to remove enrolled students
def remove_enrolled_students(df:pd.DataFrame) -> pd.DataFrame:
    return df[df['target'] != "Enrolled"]

#convert our target variable into numeric 
# "1" to represent a "Graduated" Status and "0" to represent "Dropout" status
def make_target_numeric(df:pd.DataFrame) -> pd.DataFrame:
    if is_numeric_dtype(df['target']) == False:
        df['target'] = df["target"].replace({"Graduate":1, "Dropout":0})
        return df

#combines all the cleanup methods
def clean_data() -> pd.DataFrame:
    #we first load the data
    df_students = load_data()

    #check if dataframe is empty
    if(df_students.empty):
        print("No data to clean")
        return df_students
    #rename columns
    df_students = rename_columns(df_students)

    #remove enrolled status
    df_students = remove_enrolled_students(df_students)

    #convert target variable to numeric
    df_students = make_target_numeric(df_students)

    return df_students




load_data()