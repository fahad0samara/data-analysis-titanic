"""Interfaces defining the contracts for the application."""
from abc import ABC, abstractmethod
from typing import List, Optional
from datetime import datetime
from .entities import Passenger, ModelPrediction, ModelMetrics

class IPassengerRepository(ABC):
    """Interface for passenger data repository."""
    
    @abstractmethod
    def get_all(self) -> List[Passenger]:
        """Retrieve all passengers."""
        pass
    
    @abstractmethod
    def get_by_id(self, passenger_id: int) -> Optional[Passenger]:
        """Retrieve a passenger by ID."""
        pass
    
    @abstractmethod
    def save(self, passenger: Passenger) -> None:
        """Save a passenger to the repository."""
        pass

class IModelRepository(ABC):
    """Interface for model repository."""
    
    @abstractmethod
    def save_metrics(self, metrics: ModelMetrics) -> None:
        """Save model metrics."""
        pass
    
    @abstractmethod
    def get_latest_metrics(self) -> Optional[ModelMetrics]:
        """Get the latest model metrics."""
        pass
    
    @abstractmethod
    def save_prediction(self, prediction: ModelPrediction) -> None:
        """Save a model prediction."""
        pass
    
    @abstractmethod
    def get_predictions(self, start_date: datetime, end_date: datetime) -> List[ModelPrediction]:
        """Get predictions within a date range."""
        pass

class IModelService(ABC):
    """Interface for model service."""
    
    @abstractmethod
    def train(self, data: List[Passenger]) -> ModelMetrics:
        """Train the model and return metrics."""
        pass
    
    @abstractmethod
    def predict(self, passenger: Passenger) -> ModelPrediction:
        """Make a prediction for a passenger."""
        pass
    
    @abstractmethod
    def evaluate(self, test_data: List[Passenger]) -> ModelMetrics:
        """Evaluate model performance on test data."""
        pass

class IDataPreprocessor(ABC):
    """Interface for data preprocessing."""
    
    @abstractmethod
    def preprocess(self, data: List[Passenger]) -> List[Passenger]:
        """Preprocess passenger data."""
        pass
    
    @abstractmethod
    def feature_engineering(self, data: List[Passenger]) -> List[Passenger]:
        """Perform feature engineering on passenger data."""
        pass
    
    @abstractmethod
    def encode_categorical(self, data: List[Passenger]) -> List[Passenger]:
        """Encode categorical features."""
        pass

class IDataValidator(ABC):
    """Interface for data validation."""
    
    @abstractmethod
    def validate_passenger(self, passenger: Passenger) -> bool:
        """Validate passenger data."""
        pass
    
    @abstractmethod
    def validate_prediction(self, prediction: ModelPrediction) -> bool:
        """Validate prediction data."""
        pass
    
    @abstractmethod
    def validate_metrics(self, metrics: ModelMetrics) -> bool:
        """Validate model metrics."""
        pass
