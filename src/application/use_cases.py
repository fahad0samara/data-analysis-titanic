"""Use cases for the Titanic Survival Analysis project."""
from typing import List, Dict, Optional
from datetime import datetime
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

from src.domain.entities import Passenger, ModelPrediction, ModelMetrics
from src.config.config import (
    TITANIC_PROCESSED_DATA,
    BEST_MODEL_PATH,
    MODEL_METRICS_PATH,
    MODEL_PARAMS,
    FEATURE_ENGINEERING_PARAMS
)

class TrainModelUseCase:
    """Use case for training the survival prediction model."""
    
    def execute(self, data: pd.DataFrame) -> ModelMetrics:
        """Train the model and return performance metrics."""
        # Prepare data
        X = data[FEATURE_ENGINEERING_PARAMS['numerical_features'] + 
                FEATURE_ENGINEERING_PARAMS['categorical_features']]
        y = data[FEATURE_ENGINEERING_PARAMS['target']]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        model = RandomForestClassifier(**MODEL_PARAMS)
        model.fit(X_train, y_train)
        
        # Save model
        joblib.dump(model, BEST_MODEL_PATH)
        
        # Calculate metrics
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        metrics = ModelMetrics(
            model_version=datetime.now().strftime("%Y%m%d_%H%M%S"),
            accuracy=accuracy_score(y_test, y_pred),
            precision=precision_score(y_test, y_pred),
            recall=recall_score(y_test, y_pred),
            f1_score=f1_score(y_test, y_pred),
            roc_auc=roc_auc_score(y_test, y_prob),
            training_timestamp=datetime.now(),
            feature_importance=dict(zip(X.columns, model.feature_importances_)),
            confusion_matrix=confusion_matrix(y_test, y_pred).tolist()
        )
        
        return metrics

class PredictSurvivalUseCase:
    """Use case for predicting survival probability for new passengers."""
    
    def execute(self, passenger: Passenger) -> ModelPrediction:
        """Predict survival probability for a passenger."""
        # Load model
        model = joblib.load(BEST_MODEL_PATH)
        
        # Prepare features
        features = pd.DataFrame([{
            'Pclass': passenger.pclass,
            'Sex': passenger.sex,
            'Age': passenger.age,
            'SibSp': passenger.sibsp,
            'Parch': passenger.parch,
            'Fare': passenger.fare,
            'Embarked': passenger.embarked
        }])
        
        # Make prediction
        survival_prob = model.predict_proba(features)[0, 1]
        predicted_survival = survival_prob >= 0.5
        
        prediction = ModelPrediction(
            passenger_id=passenger.passenger_id,
            survival_probability=survival_prob,
            predicted_survival=predicted_survival,
            prediction_timestamp=datetime.now(),
            model_version=datetime.fromtimestamp(BEST_MODEL_PATH.stat().st_mtime)
                                  .strftime("%Y%m%d_%H%M%S"),
            feature_importance=dict(zip(features.columns, model.feature_importances_))
        )
        
        return prediction

class AnalyzePassengersUseCase:
    """Use case for analyzing passenger data and generating insights."""
    
    def execute(self, data: pd.DataFrame) -> Dict:
        """Analyze passenger data and return insights."""
        insights = {
            'total_passengers': len(data),
            'survival_rate': data['Survived'].mean(),
            'class_distribution': data['Pclass'].value_counts().to_dict(),
            'gender_distribution': data['Sex'].value_counts().to_dict(),
            'age_statistics': {
                'mean': data['Age'].mean(),
                'median': data['Age'].median(),
                'std': data['Age'].std()
            },
            'fare_statistics': {
                'mean': data['Fare'].mean(),
                'median': data['Fare'].median(),
                'std': data['Fare'].std()
            },
            'survival_by_class': data.groupby('Pclass')['Survived'].mean().to_dict(),
            'survival_by_gender': data.groupby('Sex')['Survived'].mean().to_dict(),
            'family_size_impact': data.groupby('FamilySize')['Survived'].mean().to_dict()
        }
        
        return insights
