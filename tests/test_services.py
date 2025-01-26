"""Tests for model service."""
import pytest
import pandas as pd
import numpy as np
import os
from src.application.services import ModelService
from src.domain.entities import Passenger
from src.domain.exceptions import ModelNotFoundError, PredictionError
from src.config.config import MODEL_PATH

@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'Age': [25, 35, 45, 55, 65, 28, 38, 48],
        'Fare': [10.0, 30.0, 60.0, 120.0, 250.0, 15.0, 35.0, 70.0],
        'Survived': [0, 1, 1, 0, 1, 0, 1, 0],  # Equal number of survivors and non-survivors
        'Pclass': [1, 2, 3, 1, 2, 3, 1, 2],
        'Sex': [0, 1, 0, 1, 0, 1, 0, 1],  # 0: male, 1: female
        'SibSp': [1, 0, 2, 1, 0, 1, 0, 2],
        'Parch': [0, 2, 1, 0, 1, 0, 2, 1],
        'Embarked': [0, 1, 2, 0, 1, 2, 0, 1]  # 0: S, 1: C, 2: Q
    })

@pytest.fixture
def model_service():
    """Create a model service instance."""
    return ModelService()

@pytest.fixture(autouse=True)
def cleanup_model():
    """Clean up model file after each test."""
    yield
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)

def test_model_training(model_service, sample_df):
    """Test model training."""
    metrics = model_service.train(sample_df)
    assert metrics is not None
    assert hasattr(metrics, 'accuracy')
    assert hasattr(metrics, 'feature_importance')
    assert isinstance(metrics.accuracy, float)
    assert isinstance(metrics.feature_importance, dict)

def test_model_prediction(model_service, sample_df):
    """Test model prediction."""
    # Train the model first
    model_service.train(sample_df)
    
    # Create a test passenger
    passenger = Passenger(
        passenger_id=1,
        pclass=1,
        sex=0,  # male
        age=30,
        sibsp=1,
        parch=0,
        fare=50.0,
        embarked=0  # S
    )
    
    prediction = model_service.predict(passenger)
    assert prediction is not None
    assert hasattr(prediction, 'survival_probability')
    assert isinstance(prediction.survival_probability, float)
    assert isinstance(prediction.predicted_survival, bool)

def test_prediction_without_training(model_service):
    """Test prediction without training raises error."""
    # Ensure no model file exists
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
        
    passenger = Passenger(
        passenger_id=1,
        pclass=1,
        sex=0,
        age=30,
        sibsp=1,
        parch=0,
        fare=50.0,
        embarked=0
    )
    
    with pytest.raises(ModelNotFoundError):
        model_service.predict(passenger)

def test_model_evaluation(model_service, sample_df):
    """Test model evaluation."""
    # Train the model first
    model_service.train(sample_df)
    
    # Evaluate on test data
    metrics = model_service.evaluate(sample_df)
    assert metrics is not None
    assert 'accuracy' in metrics
    assert 'feature_importance' in metrics
    assert isinstance(metrics['accuracy'], float)
    assert isinstance(metrics['feature_importance'], dict)
