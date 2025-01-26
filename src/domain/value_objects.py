"""Value objects for the Titanic Survival Analysis project."""
from dataclasses import dataclass
from typing import List, Dict
from datetime import datetime

@dataclass(frozen=True)
class PassengerName:
    """Value object for passenger name."""
    title: str
    first_name: str
    last_name: str
    
    def __str__(self) -> str:
        return f"{self.title}. {self.first_name} {self.last_name}"

@dataclass(frozen=True)
class PassengerClass:
    """Value object for passenger class."""
    class_number: int
    description: str
    
    def __str__(self) -> str:
        return f"{self.class_number} ({self.description})"

@dataclass(frozen=True)
class FamilyInfo:
    """Value object for family information."""
    siblings_spouses: int
    parents_children: int
    family_size: int
    is_alone: bool

@dataclass(frozen=True)
class TicketInfo:
    """Value object for ticket information."""
    number: str
    fare: float
    cabin: str
    embarked: str

@dataclass(frozen=True)
class ModelVersion:
    """Value object for model version information."""
    version: str
    timestamp: datetime
    description: str
    parameters: Dict

@dataclass(frozen=True)
class FeatureSet:
    """Value object for feature set information."""
    features: List[str]
    categorical_features: List[str]
    numerical_features: List[str]
    engineered_features: List[str]

@dataclass(frozen=True)
class PredictionMetrics:
    """Value object for prediction metrics."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    timestamp: datetime

@dataclass(frozen=True)
class DatasetMetrics:
    """Value object for dataset metrics."""
    total_records: int
    missing_values: Dict[str, int]
    feature_statistics: Dict[str, Dict]
    timestamp: datetime
