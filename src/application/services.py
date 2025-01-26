"""Application services for the Titanic Survival Analysis project."""
from typing import List, Dict, Optional
from datetime import datetime
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.domain.entities import Passenger, ModelPrediction, ModelMetrics
from src.domain.interfaces import IModelService, IDataPreprocessor
from src.domain.exceptions import ModelNotFoundError, PredictionError
from src.config.config import MODEL_PARAMS, FEATURE_ENGINEERING_PARAMS

class ModelService(IModelService):
    """Service for model operations."""
    
    def __init__(self, preprocessor: IDataPreprocessor):
        self.preprocessor = preprocessor
        self.model = RandomForestClassifier(**MODEL_PARAMS)
    
    def train(self, data: List[Passenger]) -> ModelMetrics:
        """Train the model and return metrics."""
        # Preprocess data
        processed_data = self.preprocessor.preprocess(data)
        processed_data = self.preprocessor.feature_engineering(processed_data)
        processed_data = self.preprocessor.encode_categorical(processed_data)
        
        # Convert to DataFrame
        df = pd.DataFrame([vars(p) for p in processed_data])
        
        # Split features and target
        features = FEATURE_ENGINEERING_PARAMS['numerical_features'] + \
                  FEATURE_ENGINEERING_PARAMS['categorical_features']
        X = df[features]
        y = df['Survived']
        
        # Train model
        self.model.fit(X, y)
        
        # Calculate metrics
        y_pred = self.model.predict(X)
        
        return ModelMetrics(
            model_version=datetime.now().strftime("%Y%m%d_%H%M%S"),
            accuracy=accuracy_score(y, y_pred),
            precision=precision_score(y, y_pred),
            recall=recall_score(y, y_pred),
            f1_score=f1_score(y, y_pred),
            roc_auc=0.0,  # Calculate if needed
            training_timestamp=datetime.now(),
            feature_importance=dict(zip(features, self.model.feature_importances_)),
            confusion_matrix=[[0, 0], [0, 0]]  # Calculate if needed
        )
    
    def predict(self, passenger: Passenger) -> ModelPrediction:
        """Make a prediction for a passenger."""
        if not hasattr(self, 'model') or self.model is None:
            raise ModelNotFoundError("Model not trained")
        
        try:
            # Preprocess single passenger
            processed = self.preprocessor.preprocess([passenger])[0]
            processed = self.preprocessor.feature_engineering([processed])[0]
            processed = self.preprocessor.encode_categorical([processed])[0]
            
            # Convert to DataFrame
            df = pd.DataFrame([vars(processed)])
            
            # Select features
            features = FEATURE_ENGINEERING_PARAMS['numerical_features'] + \
                      FEATURE_ENGINEERING_PARAMS['categorical_features']
            X = df[features]
            
            # Make prediction
            probability = self.model.predict_proba(X)[0][1]
            prediction = probability >= 0.5
            
            return ModelPrediction(
                passenger_id=passenger.passenger_id,
                survival_probability=probability,
                predicted_survival=prediction,
                prediction_timestamp=datetime.now(),
                model_version=datetime.now().strftime("%Y%m%d_%H%M%S"),
                feature_importance=dict(zip(features, self.model.feature_importances_))
            )
            
        except Exception as e:
            raise PredictionError(f"Prediction failed: {str(e)}")
    
    def evaluate(self, test_data: List[Passenger]) -> ModelMetrics:
        """Evaluate model performance on test data."""
        if not hasattr(self, 'model') or self.model is None:
            raise ModelNotFoundError("Model not trained")
        
        # Preprocess test data
        processed_data = self.preprocessor.preprocess(test_data)
        processed_data = self.preprocessor.feature_engineering(processed_data)
        processed_data = self.preprocessor.encode_categorical(processed_data)
        
        # Convert to DataFrame
        df = pd.DataFrame([vars(p) for p in processed_data])
        
        # Split features and target
        features = FEATURE_ENGINEERING_PARAMS['numerical_features'] + \
                  FEATURE_ENGINEERING_PARAMS['categorical_features']
        X = df[features]
        y = df['Survived']
        
        # Make predictions
        y_pred = self.model.predict(X)
        
        return ModelMetrics(
            model_version=datetime.now().strftime("%Y%m%d_%H%M%S"),
            accuracy=accuracy_score(y, y_pred),
            precision=precision_score(y, y_pred),
            recall=recall_score(y, y_pred),
            f1_score=f1_score(y, y_pred),
            roc_auc=0.0,  # Calculate if needed
            training_timestamp=datetime.now(),
            feature_importance=dict(zip(features, self.model.feature_importances_)),
            confusion_matrix=[[0, 0], [0, 0]]  # Calculate if needed
        )
