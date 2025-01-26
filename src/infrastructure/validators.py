"""Data validators for the Titanic Survival Analysis project."""
from typing import Dict, Any, Optional
from datetime import datetime

from src.domain.entities import Passenger, ModelPrediction, ModelMetrics
from src.domain.interfaces import IDataValidator
from src.domain.exceptions import ValidationError

class PassengerValidator(IDataValidator):
    """Validator for passenger data."""
    
    def validate_passenger(self, passenger: Passenger) -> bool:
        """Validate passenger data."""
        try:
            # Validate required fields
            if passenger.passenger_id <= 0:
                raise ValidationError("Invalid passenger ID")
            
            if passenger.pclass not in [1, 2, 3]:
                raise ValidationError("Invalid passenger class")
            
            if not passenger.name or len(passenger.name.strip()) == 0:
                raise ValidationError("Name is required")
            
            if passenger.sex not in ['male', 'female']:
                raise ValidationError("Invalid gender")
            
            # Validate numeric fields
            if passenger.age is not None and (passenger.age < 0 or passenger.age > 120):
                raise ValidationError("Invalid age")
            
            if passenger.sibsp < 0:
                raise ValidationError("Invalid number of siblings/spouses")
            
            if passenger.parch < 0:
                raise ValidationError("Invalid number of parents/children")
            
            if passenger.fare < 0:
                raise ValidationError("Invalid fare")
            
            return True
            
        except ValidationError:
            return False
    
    def validate_prediction(self, prediction: ModelPrediction) -> bool:
        """Validate prediction data."""
        try:
            if prediction.passenger_id <= 0:
                raise ValidationError("Invalid passenger ID")
            
            if not 0 <= prediction.survival_probability <= 1:
                raise ValidationError("Invalid survival probability")
            
            if not isinstance(prediction.predicted_survival, bool):
                raise ValidationError("Invalid prediction result")
            
            if not prediction.prediction_timestamp:
                raise ValidationError("Missing prediction timestamp")
            
            if not prediction.model_version:
                raise ValidationError("Missing model version")
            
            return True
            
        except ValidationError:
            return False
    
    def validate_metrics(self, metrics: ModelMetrics) -> bool:
        """Validate model metrics."""
        try:
            # Validate metric values
            for metric_value in [metrics.accuracy, metrics.precision, 
                               metrics.recall, metrics.f1_score, metrics.roc_auc]:
                if not 0 <= metric_value <= 1:
                    raise ValidationError("Invalid metric value")
            
            # Validate timestamps
            if not metrics.training_timestamp:
                raise ValidationError("Missing training timestamp")
            
            # Validate feature importance
            if not metrics.feature_importance or not all(0 <= v <= 1 
                for v in metrics.feature_importance.values()):
                raise ValidationError("Invalid feature importance values")
            
            # Validate confusion matrix
            if not metrics.confusion_matrix or \
               len(metrics.confusion_matrix) != 2 or \
               len(metrics.confusion_matrix[0]) != 2:
                raise ValidationError("Invalid confusion matrix format")
            
            return True
            
        except ValidationError:
            return False

class DatasetValidator:
    """Validator for dataset-level validation."""
    
    @staticmethod
    def validate_schema(data: Dict[str, Any], required_fields: set) -> bool:
        """Validate that all required fields are present in the data."""
        return all(field in data for field in required_fields)
    
    @staticmethod
    def validate_data_types(data: Dict[str, Any], type_mapping: Dict[str, type]) -> bool:
        """Validate that fields have correct data types."""
        return all(isinstance(data.get(field), field_type) 
                  for field, field_type in type_mapping.items() 
                  if field in data)
    
    @staticmethod
    def validate_value_ranges(data: Dict[str, Any], range_mapping: Dict[str, tuple]) -> bool:
        """Validate that numeric fields are within specified ranges."""
        for field, (min_val, max_val) in range_mapping.items():
            value = data.get(field)
            if value is not None and not min_val <= value <= max_val:
                return False
        return True
    
    @staticmethod
    def validate_categorical_values(data: Dict[str, Any], 
                                 categorical_mapping: Dict[str, set]) -> bool:
        """Validate that categorical fields have allowed values."""
        return all(data.get(field) in allowed_values 
                  for field, allowed_values in categorical_mapping.items() 
                  if field in data)
