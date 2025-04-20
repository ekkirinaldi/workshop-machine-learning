import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import pickle

def preprocess_data(data):
    # Drop unnecessary columns
    cols_to_drop = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']
    data = data.drop(cols_to_drop, axis=1)
    
    # Convert categorical variables
    categorical_columns = data.select_dtypes(include=['object']).columns
    le = LabelEncoder()
    for col in categorical_columns:
        data[col] = le.fit_transform(data[col])
    
    return data

def train_and_save_model():
    # Load the data
    print("Loading data...")
    data = pd.read_csv('data-employee-attrition.csv')
    
    # Preprocess the data
    print("Preprocessing data...")
    processed_data = preprocess_data(data)
    
    # Separate features and target
    X = processed_data.drop('Attrition', axis=1)
    y = processed_data['Attrition']
    
    # Split the data
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train XGBoost model
    print("Training XGBoost model...")
    model = xgb.XGBClassifier(
        random_state=42,
        learning_rate=0.1,
        max_depth=5,
        n_estimators=100
    )
    model.fit(X_train, y_train)
    
    # Evaluate the model
    print("\nEvaluating model performance...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save the model and feature names
    print("\nSaving model...")
    with open('xgboost_model.pkl', 'wb') as f:
        pickle.dump({
            'model': model,
            'features': X.columns.tolist()
        }, f)
    
    print("Model trained and saved successfully as 'xgboost_model.pkl'!")
    return model

if __name__ == "__main__":
    train_and_save_model() 