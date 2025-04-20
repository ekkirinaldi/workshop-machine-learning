"""
Employee Attrition Prediction Model Training Script

This script handles the training of an XGBoost model for predicting employee attrition.
It includes data preprocessing, model training, evaluation, and model persistence.

Key Features:
- Data loading and preprocessing
- Feature engineering
- Model training with XGBoost
- Model evaluation and metrics
- Cross-platform model persistence
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import pickle
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the input data for model training.
    
    Args:
        data (pd.DataFrame): Raw input data
        
    Returns:
        pd.DataFrame: Preprocessed data ready for model training
        
    Processing Steps:
    1. Drop unnecessary columns
    2. Convert categorical variables to numerical
    3. Handle missing values (if any)
    """
    try:
        # Drop columns that don't contribute to prediction
        cols_to_drop = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']
        data = data.drop(cols_to_drop, axis=1)
        
        # Convert categorical variables to numerical
        categorical_columns = data.select_dtypes(include=['object']).columns
        le = LabelEncoder()
        for col in categorical_columns:
            data[col] = le.fit_transform(data[col])
        
        return data
    except Exception as e:
        logger.error(f"Error in data preprocessing: {str(e)}")
        raise

def train_and_save_model() -> xgb.XGBClassifier:
    """
    Train the XGBoost model and save it to disk.
    
    Returns:
        xgb.XGBClassifier: Trained XGBoost model
        
    Steps:
    1. Load and preprocess data
    2. Split into train/test sets
    3. Train XGBoost model
    4. Evaluate model performance
    5. Save model and metadata
    """
    try:
        # Load the data
        logger.info("Loading data...")
        data = pd.read_csv('data-employee-attrition.csv')
        
        # Preprocess the data
        logger.info("Preprocessing data...")
        processed_data = preprocess_data(data)
        
        # Separate features and target
        X = processed_data.drop('Attrition', axis=1)
        y = processed_data['Attrition']
        
        # Split the data
        logger.info("Splitting data into train and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.2, 
            random_state=42
        )
        
        # Train XGBoost model
        logger.info("Training XGBoost model...")
        model = xgb.XGBClassifier(
            random_state=42,
            learning_rate=0.1,
            max_depth=5,
            n_estimators=100
        )
        model.fit(X_train, y_train)
        
        # Evaluate the model
        logger.info("Evaluating model performance...")
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Model Accuracy: {accuracy:.2%}")
        logger.info("\nClassification Report:")
        logger.info(classification_report(y_test, y_pred))
        
        # Save the model and feature names
        logger.info("Saving model...")
        model_data = {
            'model': model,
            'features': X.columns.tolist()
        }
        
        # Use os.path for cross-platform compatibility
        model_path = os.path.join(os.path.dirname(__file__), 'xgboost_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info("Model trained and saved successfully!")
        return model
        
    except Exception as e:
        logger.error(f"Error in model training: {str(e)}")
        raise

if __name__ == "__main__":
    """
    Main execution block.
    Runs the model training process when the script is executed directly.
    """
    try:
        train_and_save_model()
    except Exception as e:
        logger.error(f"Script execution failed: {str(e)}")
        raise 