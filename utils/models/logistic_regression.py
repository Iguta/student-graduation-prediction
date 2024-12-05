import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle
import numpy as np

import os
import sys

# Get the absolute path of the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
# Add the project root to Python path
sys.path.append(project_root)

from utils.preprocessing import *

class LogisticRegressionModel:
    def __init__(self, max_iter=20000):
        self.model = LogisticRegression(max_iter=max_iter)
        self.scaler = StandardScaler()
        self.features = [
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

    def prepare_data(self):
        df = clean_data()
        X = df[self.features]
        y = df['target']
        return X, y

    def split_and_scale_data(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X, threshold=0.5):
        X_scaled = self.scaler.transform(X)
        y_pred_prob = self.model.predict_proba(X_scaled)[:, 1]
        y_pred = (y_pred_prob >= threshold).astype(int)
        return y_pred, y_pred_prob

    def evaluate(self, X, y, threshold=0.5):
        y_pred, _ = self.predict(X, threshold)
        return {
            'accuracy': accuracy_score(y, y_pred),
            'confusion_matrix': confusion_matrix(y, y_pred),
            'report': classification_report(y, y_pred, target_names=["Dropout", "Graduate"])
        }

    def save(self, model_path="model.pkl", scaler_path="scaler.pkl"):
        with open(model_path, 'wb') as model_file:
            pickle.dump(self.model, model_file)
        with open(scaler_path, 'wb') as scaler_file:
            pickle.dump(self.scaler, scaler_file)

    @classmethod
    def load(cls, model_path="model.pkl", scaler_path="scaler.pkl"):
        instance = cls()
        with open(model_path, 'rb') as model_file:
            instance.model = pickle.load(model_file)
        with open(scaler_path, 'rb') as scaler_file:
            instance.scaler = pickle.load(scaler_file)
        return instance

    @staticmethod
    def calculate_target_distribution(y):
        counts = y.value_counts()
        percentages = (counts / len(y)) * 100
        return percentages

def train_and_evaluate():
    model = LogisticRegressionModel()
    X, y = model.prepare_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model.scaler.fit(X_train)
    X_train_scaled = model.scaler.transform(X_train)
    X_test_scaled = model.scaler.transform(X_test)
    
    model.fit(X_train_scaled, y_train)
    
    metrics = model.evaluate(X_test_scaled, y_test)
    print(f'Accuracy: {metrics["accuracy"] * 100:.2f}%')
    print("\nConfusion Matrix:\n", metrics["confusion_matrix"])
    print("\nClassification Report:\n", metrics["report"])
    
    model.save()
    
    distribution = model.calculate_target_distribution(y)
    print("\nTarget Distribution:")
    for target, percentage in distribution.items():
        print(f"{target}: {percentage:.2f}%")
    
    return model

if __name__ == "__main__":
    train_and_evaluate()