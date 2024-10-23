import sys
import os

# Add the project root (STUDENT-GRADUATION-PREDICTION) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# print(sys.path)
from utils.preprocessing import *

df_student_cleaned = clean_data()

print(df_student_cleaned.head())