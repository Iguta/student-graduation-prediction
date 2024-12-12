import os
import sys
# # Get the absolute path of the project root directory
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
# # Add the project root to Python path
# sys.path.append(project_root)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import *
import pickle
from xgboost import XGBClassifier
import numpy as np


from utils.preprocessing import *

df_cleaned = clean_data()

# Step 2: Train XGBoost Model
def train_xgboost(X_train, y_train, **xgb_params) -> XGBClassifier:
    """
    Train an XGBoost classifier with specified parameters.
    """
    model = XGBClassifier(**xgb_params)
    model.fit(X_train, y_train)
    return model


# Step 3: Evaluate Model
def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model using accuracy, recall, precision, and F1 score.
    """
    y_pred = model.predict(X_test)
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred)
    }
    return metrics


# Step 4: Save Model
def save_model(model, filepath):
    """
    Save the trained model to a specified filepath.
    """
    with open(filepath, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved to {filepath}")

def predict(model, input_data: pd.DataFrame):
    """
    Use the trained model to make predictions on the input data.
    
    Parameters:
        model: Trained model (e.g., XGBClassifier).
        input_data: Pandas DataFrame containing input features for prediction.
        
    Returns:
        Predicted class labels and probabilities.
    """
    # Make predictions
    predicted_class = model.predict(input_data)
    predicted_probabilities = model.predict_proba(input_data)[:, 1]
    # Convert probabilities to Python floats
    predicted_probabilities = predicted_probabilities.astype(float)

    return predicted_class, predicted_probabilities


# Main Workflow
if __name__ == "__main__":
    # Load dataset
    df_cleaned = clean_data()

    # Features to use
    selected_features = [
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

    # Preprocess data
    df_selected = df_cleaned[selected_features + ['target']]

    # Split data
    X = df_selected.drop(columns=['target'])
    y = df_selected['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model with explicit parameters
    xgb_params = {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42
    }
    model = train_xgboost(X_train, y_train, **xgb_params)

    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.2f}")

    # Save model
    save_model(model, './xgb_model.pkl')