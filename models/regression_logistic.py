import sys
import os

# Add the project root (STUDENT-GRADUATION-PREDICTION) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
#importing our python submodule
from utils.preprocessing import *

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report




df_student_cleaned = clean_data()

#select features for our logistic model
features =  [
    "2nd_sem_units_approved", 
    "2nd_sem_units_grade", 
    "1st_sem_units_approved", 
    "1st_sem_units_grade", 
    "course",
    "tuition_fees_up_to_date", 
    "scholarship_holder",
    "enrollment_age",
    "gender",
    "marital_status"           
]


#define feature matrix and target variable
X =  df_student_cleaned[features]
y =  df_student_cleaned['target']

#split the data into training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#initialize and train the logistic model
model = LogisticRegression(max_iter=20000)
model.fit(X_train, y_train)

#make predictions of test data
y_pred = model.predict(X_test)

#evaluate the model
accuracy = accuracy_score(y_test, y_pred)
confusion_matrix = confusion_matrix(y_test, y_pred)

#evaluate precision, recall, and F1-score
report = classification_report(y_test, y_pred, target_names=["Dropout", "Graduate"])

#print the accuracy and confusion matrix
print(f'Accuracy:{accuracy *100:.2f}%')
print("Confusion Matrix")
print(confusion_matrix)

print(report)
