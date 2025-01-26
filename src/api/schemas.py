"""API schemas for request and response validation."""
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional
from datetime import datetime

class PassengerBase(BaseModel):
    """Base schema for passenger data."""
    passenger_id: int = Field(..., gt=0, description="Unique passenger ID")
    pclass: int = Field(..., ge=1, le=3, description="Passenger class")
    name: str = Field(..., min_length=1, description="Passenger name")
    sex: str = Field(..., regex="^(male|female)$", description="Passenger gender")
    age: Optional[float] = Field(None, ge=0, le=120, description="Passenger age")
    sibsp: int = Field(..., ge=0, description="Number of siblings/spouses aboard")
    parch: int = Field(..., ge=0, description="Number of parents/children aboard")
    ticket: str = Field(..., description="Ticket number")
    fare: float = Field(..., ge=0, description="Passenger fare")
    cabin: Optional[str] = Field(None, description="Cabin number")
    embarked: Optional[str] = Field(None, regex="^[CQS]$", description="Port of embarkation")

    class Config:
        schema_extra = {
            "example": {
                "passenger_id": 1,
                "pclass": 3,
                "name": "Braund, Mr. Owen Harris",
                "sex": "male",
                "age": 22.0,
                "sibsp": 1,
                "parch": 0,
                "ticket": "A/5 21171",
                "fare": 7.25,
                "cabin": None,
                "embarked": "S"
            }
        }

class PassengerCreate(PassengerBase):
    """Schema for creating a new passenger."""
    pass

class PassengerResponse(PassengerBase):
    """Schema for passenger response."""
    survived: Optional[bool] = Field(None, description="Survival status")

    class Config:
        schema_extra = {
            "example": {
                "passenger_id": 1,
                "pclass": 3,
                "name": "Braund, Mr. Owen Harris",
                "sex": "male",
                "age": 22.0,
                "sibsp": 1,
                "parch": 0,
                "ticket": "A/5 21171",
                "fare": 7.25,
                "cabin": None,
                "embarked": "S",
                "survived": False
            }
        }

class PredictionRequest(BaseModel):
    """Schema for prediction request."""
    passenger: PassengerBase

class PredictionResponse(BaseModel):
    """Schema for prediction response."""
    passenger_id: int = Field(..., description="Passenger ID")
    survival_probability: float = Field(..., ge=0, le=1, description="Probability of survival")
    predicted_survival: bool = Field(..., description="Predicted survival status")
    prediction_timestamp: datetime = Field(..., description="Timestamp of prediction")
    model_version: str = Field(..., description="Version of the model used")
    feature_importance: Dict[str, float] = Field(..., description="Feature importance scores")

    class Config:
        schema_extra = {
            "example": {
                "passenger_id": 1,
                "survival_probability": 0.32,
                "predicted_survival": False,
                "prediction_timestamp": "2025-01-26T10:41:35+02:00",
                "model_version": "20250126_104135",
                "feature_importance": {
                    "age": 0.15,
                    "fare": 0.25,
                    "pclass": 0.3
                }
            }
        }

class ModelMetricsResponse(BaseModel):
    """Schema for model metrics response."""
    model_version: str = Field(..., description="Model version")
    accuracy: float = Field(..., ge=0, le=1, description="Model accuracy")
    precision: float = Field(..., ge=0, le=1, description="Model precision")
    recall: float = Field(..., ge=0, le=1, description="Model recall")
    f1_score: float = Field(..., ge=0, le=1, description="Model F1 score")
    roc_auc: float = Field(..., ge=0, le=1, description="ROC AUC score")
    training_timestamp: datetime = Field(..., description="Model training timestamp")
    feature_importance: Dict[str, float] = Field(..., description="Feature importance scores")
    confusion_matrix: List[List[int]] = Field(..., description="Confusion matrix")

    class Config:
        schema_extra = {
            "example": {
                "model_version": "20250126_104135",
                "accuracy": 0.85,
                "precision": 0.83,
                "recall": 0.87,
                "f1_score": 0.85,
                "roc_auc": 0.89,
                "training_timestamp": "2025-01-26T10:41:35+02:00",
                "feature_importance": {
                    "age": 0.15,
                    "fare": 0.25,
                    "pclass": 0.3
                },
                "confusion_matrix": [[100, 20], [15, 85]]
            }
        }

class ErrorResponse(BaseModel):
    """Schema for error response."""
    error: str = Field(..., description="Error message")
    error_code: str = Field(..., description="Error code")
    timestamp: datetime = Field(..., description="Error timestamp")

    class Config:
        schema_extra = {
            "example": {
                "error": "Invalid passenger data",
                "error_code": "VALIDATION_ERROR",
                "timestamp": "2025-01-26T10:41:35+02:00"
            }
        }
