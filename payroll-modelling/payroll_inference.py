"""
Payroll Inference API

This FastAPI application provides endpoints for making salary predictions using a trained model.
It includes model loading, prediction endpoints, and health checks.

Features:
- FastAPI REST API
- Model loading and validation
- Input validation using Pydantic
- Error handling and logging
- Health check endpoint

Usage:
    uvicorn payroll_inference:app --host 0.0.0.0 --port 8000

Endpoints:
- POST /predict: Make salary predictions
- GET /health: Check API and model health
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import os
from pathlib import Path

# Initialize FastAPI app
app = FastAPI(
    title="Payroll Prediction API",
    description="API for predicting annual salary based on quarterly payments",
    version="1.0.0"
)

# Get the current directory and model path
current_dir = Path(__file__).parent.absolute()
MODEL_PATH = current_dir / 'payroll_model.pkl'

def load_model():
    """
    Load the trained model from disk.
    
    Returns:
        object: The loaded model
    
    Raises:
        FileNotFoundError: If model file is not found
        Exception: For other loading errors
    """
    try:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file {MODEL_PATH} not found. Please run training script first: python payroll_train.py"
            )
        
        with open(MODEL_PATH, 'rb') as f:
            return pickle.load(f)
            
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")

# Load model at startup
try:
    model = load_model()
except Exception as e:
    print(f"Warning: {str(e)}")
    model = None

class PayrollInput(BaseModel):
    """
    Input data model for salary prediction.
    
    Attributes:
        q1_payments (float): Q1 payment amount
        q2_payments (float): Q2 payment amount
        q3_payments (float): Q3 payment amount
        q4_payments (float): Q4 payment amount
    """
    q1_payments: float
    q2_payments: float
    q3_payments: float
    q4_payments: float

class PayrollPrediction(BaseModel):
    """
    Output data model for salary prediction.
    
    Attributes:
        projected_annual_salary (float): Predicted annual salary
    """
    projected_annual_salary: float

@app.post("/predict", response_model=PayrollPrediction)
async def predict_salary(input_data: PayrollInput):
    """
    Make a salary prediction based on quarterly payments.
    
    Args:
        input_data (PayrollInput): Quarterly payment data
    
    Returns:
        PayrollPrediction: Predicted annual salary
    
    Raises:
        HTTPException: If model is not loaded or prediction fails
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure the model is trained and accessible."
        )
        
    try:
        # Convert input data to numpy array
        features = np.array([[
            input_data.q1_payments,
            input_data.q2_payments,
            input_data.q3_payments,
            input_data.q4_payments
        ]])
        
        # Make prediction
        prediction = float(model.predict(features)[0])
        
        return PayrollPrediction(projected_annual_salary=prediction)
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error making prediction: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """
    Check if the service and model are healthy.
    
    Returns:
        dict: Health status information
    
    Raises:
        HTTPException: If model is not loaded
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure the model is trained and accessible."
        )
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_path": str(MODEL_PATH)
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv('PORT', 8000))
    host = os.getenv('HOST', '0.0.0.0')
    uvicorn.run(app, host=host, port=port)
