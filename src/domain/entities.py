"""Domain entities for the Titanic Survival Analysis project."""
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

@dataclass
class Passenger:
    """Passenger entity."""
    passenger_id: int
    pclass: int
    sex: int  # 0: male, 1: female
    age: float
    sibsp: int
    parch: int
    fare: float
    embarked: int  # 0: S, 1: C, 2: Q
    name: Optional[str] = None
    ticket: Optional[str] = None
    cabin: Optional[str] = None

@dataclass
class ModelPrediction:
    """Model prediction result."""
    passenger_id: int
    survival_probability: float
    predicted_survival: bool
    prediction_timestamp: datetime = datetime.now()

@dataclass
class ModelMetrics:
    """Model evaluation metrics."""
    accuracy: float
    feature_importance: Dict[str, float]
    model_version: str = datetime.now().strftime("%Y%m%d_%H%M%S")
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    roc_auc: Optional[float] = None
    training_timestamp: datetime = datetime.now()
    confusion_matrix: Optional[List[List[int]]] = None
