from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import uvicorn
from typing import List, Optional
import numpy as np
from src.data_processing import DataProcessor
from src.features import FeatureEngineering

app = FastAPI(
    title="Titanic Survival Prediction API",
    description="API for predicting survival on the Titanic",
    version="1.0.0"
)

class PredictionInput(BaseModel):
    """Input data model for predictions"""
    pclass: int
    sex: str
    age: float
    sibsp: int
    parch: int
    fare: float
    
    class Config:
        schema_extra = {
            "example": {
                "pclass": 1,
                "sex": "female",
                "age": 29.0,
                "sibsp": 0,
                "parch": 0,
                "fare": 211.3375
            }
        }

class PredictionOutput(BaseModel):
    """Output data model for predictions"""
    survival_probability: float
    would_survive: bool
    confidence: float

class BatchPredictionInput(BaseModel):
    """Input data model for batch predictions"""
    passengers: List[PredictionInput]

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to the Titanic Survival Prediction API",
        "docs_url": "/docs"
    }

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    """
    Make a single prediction
    """
    try:
        # Load the model
        model = joblib.load('models/best_model.joblib')
        
        # Create a DataFrame from input
        df = pd.DataFrame([{
            'Pclass': input_data.pclass,
            'Sex': input_data.sex,
            'Age': input_data.age,
            'SibSp': input_data.sibsp,
            'Parch': input_data.parch,
            'Fare': input_data.fare
        }])
        
        # Make prediction
        probability = model.predict_proba(df)[0]
        prediction = model.predict(df)[0]
        
        return PredictionOutput(
            survival_probability=float(probability[1]),
            would_survive=bool(prediction),
            confidence=float(max(probability))
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch-predict")
async def batch_predict(input_data: BatchPredictionInput):
    """
    Make predictions for multiple passengers
    """
    try:
        # Load the model
        model = joblib.load('models/best_model.joblib')
        
        # Create DataFrame from input
        records = []
        for passenger in input_data.passengers:
            records.append({
                'Pclass': passenger.pclass,
                'Sex': passenger.sex,
                'Age': passenger.age,
                'SibSp': passenger.sibsp,
                'Parch': passenger.parch,
                'Fare': passenger.fare
            })
        
        df = pd.DataFrame(records)
        
        # Make predictions
        probabilities = model.predict_proba(df)
        predictions = model.predict(df)
        
        # Prepare response
        results = []
        for prob, pred in zip(probabilities, predictions):
            results.append({
                'survival_probability': float(prob[1]),
                'would_survive': bool(pred),
                'confidence': float(max(prob))
            })
            
        return {'predictions': results}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-info")
async def model_info():
    """
    Get information about the model
    """
    try:
        model = joblib.load('models/best_model.joblib')
        return {
            "model_type": type(model).__name__,
            "feature_names": model.feature_names_in_.tolist(),
            "n_features": len(model.feature_names_in_),
            "parameters": model.get_params()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
