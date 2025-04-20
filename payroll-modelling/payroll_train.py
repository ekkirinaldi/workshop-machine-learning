"""
Payroll Training Script

This script trains a linear regression model to predict annual salary based on quarterly payments.
It handles data loading, preprocessing, model training, and model saving.

Features:
- Cross-platform file path handling
- Data validation and cleaning
- Linear regression model training
- Model performance evaluation
- Error handling and logging

Usage:
    python payroll_train.py

The script will:
1. Load training data from data-payroll.csv
2. Clean and preprocess the data
3. Train a linear regression model
4. Save the model to payroll_model.pkl
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import os
from pathlib import Path

def train_model():
    """
    Train a linear regression model for payroll prediction.
    
    Returns:
        bool: True if training was successful, False otherwise.
    
    Raises:
        FileNotFoundError: If training data file is not found
        ValueError: If data processing fails
        Exception: For other unexpected errors
    """
    # Get the current directory for cross-platform compatibility
    current_dir = Path(__file__).parent.absolute()
    
    # Define paths using Path for cross-platform compatibility
    data_path = current_dir / 'data-payroll.csv'
    model_path = current_dir / 'payroll_model.pkl'
    
    try:
        # Load and validate data
        if not data_path.exists():
            raise FileNotFoundError(f"Training data not found at {data_path}")
            
        df_data = pd.read_csv(data_path)
        
        # Clean data - remove $ signs and convert to float
        for item in df_data.columns[df_data.dtypes=='object']:
            df_data[item] = df_data[item].str.replace('$','')
        
        # Convert salary and payment columns to float
        numeric_columns = [
            'Projected Annual Salary',
            'Q1 Payments',
            'Q2 Payments',
            'Q3 Payments',
            'Q4 Payments'
        ]
        
        for col in numeric_columns:
            df_data[col] = pd.to_numeric(df_data[col], errors='coerce')
        
        # Remove any rows with NaN values
        df_data = df_data.dropna(subset=numeric_columns)
        
        # Prepare features and target
        df = df_data[numeric_columns]
        df.columns = ['Projected_Annual_Salary', 'Q1_Payments', 'Q2_Payments', 'Q3_Payments', 'Q4_Payments']

        # Split features and target
        X = df.drop(['Projected_Annual_Salary'], axis=1)
        y = df['Projected_Annual_Salary']

        # Split train/test data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Get model score
        score = model.score(X_test, y_test)
        print(f"Model R2 Score: {score:.4f}")

        # Save model
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        print(f"Model trained and saved as {model_path}")
        return True
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        return False

if __name__ == "__main__":
    success = train_model()
    if not success:
        exit(1)
