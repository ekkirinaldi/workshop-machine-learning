from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import os
from pathlib import Path

app = FastAPI()

# Get the current directory and model path
current_dir = Path(__file__).parent.absolute()
MODEL_PATH = current_dir / 'payroll_model.pkl'

def load_model():
    """Load the trained model from disk"""
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
    q1_payments: float
    q2_payments: float
    q3_payments: float
    q4_payments: float

class PayrollPrediction(BaseModel):
    projected_annual_salary: float

@app.post("/predict", response_model=PayrollPrediction)
async def predict_salary(input_data: PayrollInput):
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
    """Check if the service and model are healthy"""
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
