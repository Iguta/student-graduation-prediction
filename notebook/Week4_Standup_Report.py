import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the data
file_path = r"C:/Intro_DataScience/Week 4/Academic_Success_Data.csv"
data = pd.read_csv(file_path)

plt.figure(figsize=(10, 6))
sns.scatterplot(x='Curricular units 1st sem (approved)', y='Admission grade', hue='Target', data=data)
plt.title('Approved Curricular Units in 1st Sem vs Academic Outcome')
plt.xlabel('Approved Curricular Units in 1st Semester')
plt.ylabel('Admission Grade')
plt.show()

plt.figure(figsize=(8, 6))
sns.countplot(x='Gender', hue='Target', data=data)
plt.title('Gender vs Academic Outcome')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()
