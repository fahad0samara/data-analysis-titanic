"""Data repositories for the Titanic Survival Analysis project."""
import json
from pathlib import Path
from typing import List, Optional, Dict
import pandas as pd
from datetime import datetime

from src.domain.entities import Passenger, ModelMetrics
from src.config.config import (
    TITANIC_RAW_DATA,
    TITANIC_PROCESSED_DATA,
    MODEL_METRICS_PATH
)

class PassengerRepository:
    """Repository for managing passenger data."""
    
    def __init__(self, data_path: Path = TITANIC_PROCESSED_DATA):
        self.data_path = data_path
    
    def get_all(self) -> List[Passenger]:
        """Retrieve all passengers."""
        df = pd.read_csv(self.data_path)
        return [self._to_entity(row) for _, row in df.iterrows()]
    
    def get_by_id(self, passenger_id: int) -> Optional[Passenger]:
        """Retrieve a passenger by ID."""
        df = pd.read_csv(self.data_path)
        passenger_data = df[df['PassengerId'] == passenger_id]
        if len(passenger_data) == 0:
            return None
        return self._to_entity(passenger_data.iloc[0])
    
    def save(self, passenger: Passenger) -> None:
        """Save a passenger to the dataset."""
        df = pd.read_csv(self.data_path)
        passenger_dict = self._to_dict(passenger)
        
        if passenger.passenger_id in df['PassengerId'].values:
            df.loc[df['PassengerId'] == passenger.passenger_id] = passenger_dict
        else:
            df = df.append(passenger_dict, ignore_index=True)
        
        df.to_csv(self.data_path, index=False)
    
    def _to_entity(self, row: pd.Series) -> Passenger:
        """Convert a pandas Series to a Passenger entity."""
        return Passenger(
            passenger_id=row['PassengerId'],
            pclass=row['Pclass'],
            name=row['Name'],
            sex=row['Sex'],
            age=row['Age'],
            sibsp=row['SibSp'],
            parch=row['Parch'],
            ticket=row['Ticket'],
            fare=row['Fare'],
            cabin=row['Cabin'],
            embarked=row['Embarked'],
            survived=row['Survived'] if 'Survived' in row else None
        )
    
    def _to_dict(self, passenger: Passenger) -> Dict:
        """Convert a Passenger entity to a dictionary."""
        return {
            'PassengerId': passenger.passenger_id,
            'Pclass': passenger.pclass,
            'Name': passenger.name,
            'Sex': passenger.sex,
            'Age': passenger.age,
            'SibSp': passenger.sibsp,
            'Parch': passenger.parch,
            'Ticket': passenger.ticket,
            'Fare': passenger.fare,
            'Cabin': passenger.cabin,
            'Embarked': passenger.embarked,
            'Survived': passenger.survived
        }

class ModelMetricsRepository:
    """Repository for managing model metrics."""
    
    def __init__(self, metrics_path: Path = MODEL_METRICS_PATH):
        self.metrics_path = metrics_path
    
    def save(self, metrics: ModelMetrics) -> None:
        """Save model metrics."""
        metrics_dict = {
            'model_version': metrics.model_version,
            'accuracy': metrics.accuracy,
            'precision': metrics.precision,
            'recall': metrics.recall,
            'f1_score': metrics.f1_score,
            'roc_auc': metrics.roc_auc,
            'training_timestamp': metrics.training_timestamp.isoformat(),
            'feature_importance': metrics.feature_importance,
            'confusion_matrix': metrics.confusion_matrix
        }
        
        if self.metrics_path.exists():
            with open(self.metrics_path, 'r') as f:
                all_metrics = json.load(f)
        else:
            all_metrics = []
        
        all_metrics.append(metrics_dict)
        
        with open(self.metrics_path, 'w') as f:
            json.dump(all_metrics, f, indent=4)
    
    def get_latest(self) -> Optional[ModelMetrics]:
        """Retrieve the latest model metrics."""
        if not self.metrics_path.exists():
            return None
        
        with open(self.metrics_path, 'r') as f:
            all_metrics = json.load(f)
        
        if not all_metrics:
            return None
        
        latest = max(all_metrics, key=lambda x: x['training_timestamp'])
        return ModelMetrics(
            model_version=latest['model_version'],
            accuracy=latest['accuracy'],
            precision=latest['precision'],
            recall=latest['recall'],
            f1_score=latest['f1_score'],
            roc_auc=latest['roc_auc'],
            training_timestamp=datetime.fromisoformat(latest['training_timestamp']),
            feature_importance=latest['feature_importance'],
            confusion_matrix=latest['confusion_matrix']
        )
    
    def get_all(self) -> List[ModelMetrics]:
        """Retrieve all model metrics."""
        if not self.metrics_path.exists():
            return []
        
        with open(self.metrics_path, 'r') as f:
            all_metrics = json.load(f)
        
        return [
            ModelMetrics(
                model_version=m['model_version'],
                accuracy=m['accuracy'],
                precision=m['precision'],
                recall=m['recall'],
                f1_score=m['f1_score'],
                roc_auc=m['roc_auc'],
                training_timestamp=datetime.fromisoformat(m['training_timestamp']),
                feature_importance=m['feature_importance'],
                confusion_matrix=m['confusion_matrix']
            )
            for m in all_metrics
        ]
