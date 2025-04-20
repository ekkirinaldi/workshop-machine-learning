"""
Employee Attrition Prediction Inference API

This script implements a FastAPI service for making predictions using the trained XGBoost model.
It provides a RESTful API interface for real-time predictions and model health checks.

Key Features:
- FastAPI RESTful endpoints
- Real-time predictions
- Input validation
- Error handling
- Health check endpoint
- API documentation
- Cross-platform compatibility
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle
from fastapi import FastAPI, HTTPException
import uvicorn
from pydantic import BaseModel
import os
from typing import Dict, Any
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app with metadata
app = FastAPI(
    title="Employee Attrition Prediction API",
    description="API for predicting employee attrition using XGBoost model",
    version="1.0.0"
)

def get_model_path() -> str:
    """
    Get the absolute path to the model file.
    
    Returns:
        str: Absolute path to the model file
        
    Note:
        Uses os.path for cross-platform compatibility
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, 'xgboost_model.pkl')

def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess input data for prediction.
    
    Args:
        data (pd.DataFrame): Input data for prediction
        
    Returns:
        pd.DataFrame: Preprocessed data ready for prediction
        
    Processing Steps:
    1. Drop unnecessary columns
    2. Convert categorical variables
    3. Ensure feature order matches training
    """
    try:
        # Drop unnecessary columns if they exist
        cols_to_drop = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']
        for col in cols_to_drop:
            if col in data.columns:
                data = data.drop(col, axis=1)
        
        # Convert categorical variables
        categorical_columns = data.select_dtypes(include=['object']).columns
        le = LabelEncoder()
        for col in categorical_columns:
            data[col] = le.fit_transform(data[col])
        
        return data
    except Exception as e:
        logger.error(f"Error in data preprocessing: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Data preprocessing error: {str(e)}"
        )

class PredictionInput(BaseModel):
    """
    Pydantic model for input validation.
    Defines the expected structure and types of prediction input data.
    """
    Age: int
    BusinessTravel: str
    DailyRate: int
    Department: str
    DistanceFromHome: int
    Education: int
    EducationField: str
    EnvironmentSatisfaction: int
    Gender: str
    HourlyRate: int
    JobInvolvement: int
    JobLevel: int
    JobRole: str
    JobSatisfaction: int
    MaritalStatus: str
    MonthlyIncome: int
    MonthlyRate: int
    NumCompaniesWorked: int
    OverTime: str
    PercentSalaryHike: int
    PerformanceRating: int
    RelationshipSatisfaction: int
    StockOptionLevel: int
    TotalWorkingYears: int
    TrainingTimesLastYear: int
    WorkLifeBalance: int
    YearsAtCompany: int
    YearsInCurrentRole: int
    YearsSinceLastPromotion: int
    YearsWithCurrManager: int

@app.get("/")
async def root() -> Dict[str, Any]:
    """
    Root endpoint for API health check and information.
    
    Returns:
        Dict[str, Any]: API status and information
        
    Checks:
    1. API running status
    2. Model availability
    3. Available endpoints
    """
    try:
        model_path = get_model_path()
        model_status = "available" if os.path.exists(model_path) else "not available"
        return {
            "status": "API is running",
            "model_status": model_status,
            "endpoints": {
                "root": "GET /",
                "predict": "POST /predict",
                "docs": "GET /docs"
            },
            "message": "Use POST /predict for attrition predictions or visit /docs for API documentation"
        }
    except Exception as e:
        logger.error(f"Error in root endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"API error: {str(e)}"
        )

@app.post("/predict")
async def predict(input_data: PredictionInput) -> Dict[str, Any]:
    """
    Make attrition predictions using the trained XGBoost model.
    
    Args:
        input_data (PredictionInput): Input data for prediction
        
    Returns:
        Dict[str, Any]: Prediction results including:
            - probability_of_attrition
            - prediction
            - confidence
        
    Steps:
    1. Load model
    2. Preprocess input
    3. Make prediction
    4. Return results
    """
    try:
        model_path = get_model_path()
        # Check if model exists
        if not os.path.exists(model_path):
            raise HTTPException(
                status_code=503,
                detail="Model file not found. Please train the model first using model_train.py"
            )

        # Load the model
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
                model = model_data['model']
                features = model_data['features']
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error loading model: {str(e)}"
            )
        
        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data.dict()])
        processed_input = preprocess_data(input_df)
        
        # Verify all required features are present
        missing_features = set(features) - set(processed_input.columns)
        if missing_features:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required features: {missing_features}"
            )
        
        # Make prediction
        try:
            prediction = model.predict_proba(processed_input[features])[0]
            
            return {
                "status": "success",
                "probability_of_attrition": float(prediction[1]),
                "prediction": "Yes" if prediction[1] > 0.5 else "No",
                "confidence": float(max(prediction[0], prediction[1]))
            }
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Prediction error: {str(e)}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in prediction: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )

def start_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    """
    Start the FastAPI server.
    
    Args:
        host (str): Host address to bind to
        port (int): Port number to listen on
        
    Note:
        Uses uvicorn for ASGI server implementation
    """
    try:
        logger.info(f"Starting server on {host}:{port}")
        uvicorn.run(app, host=host, port=port)
    except Exception as e:
        logger.error(f"Error starting server: {str(e)}")
        raise

if __name__ == "__main__":
    """
    Main execution block.
    Starts the FastAPI server with configuration from environment variables.
    """
    try:
        # Get port from environment variable or use default
        port = int(os.getenv("PORT", 8000))
        start_server(port=port)
    except Exception as e:
        logger.error(f"Server startup failed: {str(e)}")
        raise
