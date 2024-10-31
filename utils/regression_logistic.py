import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from preprocessing import *
import pickle

def prepare_data():
    df_student_cleaned = clean_data()
    
    # Select features
    features = [
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
    X = df_student_cleaned[features]
    y = df_student_cleaned['target']
    
    return X, y

def split_and_scale_data(X, y):
    # Split the data into training and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_logistic_model(X_train_scaled, y_train):
    model = LogisticRegression(max_iter=20000)
    model.fit(X_train_scaled, y_train)
    return model

def evaluate_model(model, X_test_scaled, y_test):
    y_pred = model.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Dropout", "Graduate"])
    
    return accuracy, conf_matrix, report

def adjust_threshold(model, X_test_scaled, y_test, threshold=0.6):
    y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]
    y_pred_adjusted = (y_pred_prob >= threshold).astype(int)
    adjusted_report = classification_report(y_test, y_pred_adjusted, target_names=["Dropout", "Graduate"])
    return adjusted_report

#save model
def save_model(model, scaler, model_path="./logistic_model.pkl", scaler_path="./scaler.pkl"):
    #save trained model
    with open(model_path, 'wb') as model_file:
        pickle.dump(model, model_file)
    #save scaler
    with open(scaler_path, 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)

# Main execution
if __name__ == "__main__":
    # Prepare data
    X, y = prepare_data()
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = split_and_scale_data(X, y)
    
    # Train model
    model = train_logistic_model(X_train_scaled, y_train)
    
    # Evaluate model
    accuracy, conf_matrix, report = evaluate_model(model, X_test_scaled, y_test)
    print(f'Accuracy: {accuracy * 100:.2f}%')
    print("Confusion Matrix:\n", conf_matrix)
    print("Classification Report:\n", report)
    
    # Save model and scaler to files
    save_model(model, scaler)