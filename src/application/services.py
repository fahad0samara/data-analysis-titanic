"""Services for model training and prediction."""
from typing import Dict, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib

from src.domain.entities import Passenger, ModelPrediction, ModelMetrics
from src.domain.exceptions import ModelNotFoundError, PredictionError
from src.utils.data_utils import get_feature_columns
from src.config.config import MODEL_PARAMS, MODEL_PATH

class ModelService:
    """Service for training and using the survival prediction model."""
    
    def __init__(self):
        """Initialize the model service."""
        self.model: Optional[RandomForestClassifier] = None
        self.feature_columns = get_feature_columns()
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the data for model training or prediction."""
        # Ensure all required columns are present
        for col in self.feature_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Convert categorical variables if they're not already numeric
        if df['Sex'].dtype == 'object':
            df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
        if df['Embarked'].dtype == 'object':
            df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
        
        # Fill missing values
        df['Age'].fillna(df['Age'].median(), inplace=True)
        df['Fare'].fillna(df['Fare'].median(), inplace=True)
        df['Embarked'].fillna(0, inplace=True)  # Default to 'S'
        
        return df[self.feature_columns]
    
    def train(self, df: pd.DataFrame) -> ModelMetrics:
        """Train the model and return metrics."""
        # Prepare data
        X = self.preprocess_data(df)
        y = df['Survived']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model = RandomForestClassifier(**MODEL_PARAMS)
        self.model.fit(X_train, y_train)
        
        # Save model
        joblib.dump(self.model, MODEL_PATH)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = ModelMetrics(
            accuracy=accuracy_score(y_test, y_pred),
            precision=precision_score(y_test, y_pred),
            recall=recall_score(y_test, y_pred),
            f1_score=f1_score(y_test, y_pred),
            roc_auc=roc_auc_score(y_test, y_pred_proba),
            feature_importance=dict(zip(self.feature_columns, 
                                      self.model.feature_importances_)),
            confusion_matrix=confusion_matrix(y_test, y_pred).tolist()
        )
        
        return metrics
    
    def predict(self, passenger: Passenger) -> ModelPrediction:
        """Make a prediction for a single passenger."""
        if self.model is None:
            try:
                self.model = joblib.load(MODEL_PATH)
            except FileNotFoundError:
                raise ModelNotFoundError("Model not found. Please train the model first.")
        
        # Convert passenger to DataFrame
        passenger_df = pd.DataFrame([{
            'Age': passenger.age,
            'Fare': passenger.fare,
            'SibSp': passenger.sibsp,
            'Parch': passenger.parch,
            'Sex': passenger.sex,
            'Embarked': passenger.embarked,
            'Pclass': passenger.pclass
        }])
        
        try:
            # Preprocess data
            X = self.preprocess_data(passenger_df)
            
            # Make prediction
            survival_prob = self.model.predict_proba(X)[0][1]
            survived = survival_prob >= 0.5
            
            return ModelPrediction(
                passenger_id=passenger.passenger_id,
                survival_probability=float(survival_prob),
                predicted_survival=bool(survived)
            )
        except Exception as e:
            raise PredictionError(f"Prediction failed: {str(e)}")
    
    def evaluate(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """Evaluate the model on test data."""
        if self.model is None:
            raise ModelNotFoundError("Model not found. Please train the model first.")
        
        X_test = self.preprocess_data(test_data)
        y_test = test_data['Survived']
        
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]
        
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'feature_importance': dict(zip(self.feature_columns, 
                                         self.model.feature_importances_))
        }
