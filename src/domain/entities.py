"""Domain entities for the Titanic Survival Analysis project."""
from dataclasses import dataclass
from typing import Optional, List, Dict
from datetime import datetime

@dataclass
class Passenger:
    """Passenger entity representing a Titanic passenger."""
    passenger_id: int
    pclass: int
    name: str
    sex: str
    age: Optional[float]
    sibsp: int
    parch: int
    ticket: str
    fare: float
    cabin: Optional[str]
    embarked: Optional[str]
    survived: Optional[bool] = None

    @property
    def family_size(self) -> int:
        """Calculate total family size including the passenger."""
        return self.sibsp + self.parch + 1

@dataclass
class ModelPrediction:
    """Entity representing a model's prediction."""
    passenger_id: int
    survival_probability: float
    predicted_survival: bool
    prediction_timestamp: datetime
    model_version: str
    feature_importance: Dict[str, float]

@dataclass
class ModelMetrics:
    """Entity representing model performance metrics."""
    model_version: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    training_timestamp: datetime
    feature_importance: Dict[str, float]
    confusion_matrix: List[List[int]]
