"""Validators for domain entities."""
from typing import Optional
from src.domain.entities import Passenger, ModelPrediction

def validate_passenger(passenger: Passenger) -> Optional[None]:
    """Validate passenger data."""
    if passenger.passenger_id <= 0:
        raise ValueError("Passenger ID must be positive")
    
    if passenger.pclass not in [1, 2, 3]:
        raise ValueError("Passenger class must be 1, 2, or 3")
    
    if passenger.sex not in [0, 1]:  # 0: male, 1: female
        raise ValueError("Sex must be 0 (male) or 1 (female)")
    
    if passenger.age < 0:
        raise ValueError("Age cannot be negative")
    
    if passenger.fare < 0:
        raise ValueError("Fare cannot be negative")
    
    if passenger.sibsp < 0:
        raise ValueError("Number of siblings/spouses cannot be negative")
    
    if passenger.parch < 0:
        raise ValueError("Number of parents/children cannot be negative")
    
    if passenger.embarked not in [0, 1, 2]:  # 0: S, 1: C, 2: Q
        raise ValueError("Embarked must be 0 (S), 1 (C), or 2 (Q)")
    
    return None

def validate_prediction(prediction: ModelPrediction) -> Optional[None]:
    """Validate model prediction."""
    if prediction.passenger_id <= 0:
        raise ValueError("Passenger ID must be positive")
    
    if not 0 <= prediction.survival_probability <= 1:
        raise ValueError("Survival probability must be between 0 and 1")
    
    if not isinstance(prediction.predicted_survival, bool):
        raise ValueError("Predicted survival must be a boolean")
    
    return None
