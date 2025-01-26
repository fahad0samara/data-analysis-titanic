"""Tests for application services."""
import pytest
from datetime import datetime
from src.domain.entities import Passenger, ModelPrediction, ModelMetrics
from src.application.services import ModelService
from src.domain.interfaces import IDataPreprocessor
from src.domain.exceptions import ModelNotFoundError, PredictionError

class MockPreprocessor(IDataPreprocessor):
    """Mock preprocessor for testing."""
    
    def preprocess(self, data):
        return data
    
    def feature_engineering(self, data):
        return data
    
    def encode_categorical(self, data):
        return data

@pytest.fixture
def model_service():
    """Create a model service instance for testing."""
    return ModelService(MockPreprocessor())

@pytest.fixture
def sample_passenger():
    """Create a sample passenger for testing."""
    return Passenger(
        passenger_id=1,
        pclass=3,
        name="Test Passenger",
        sex="male",
        age=25,
        sibsp=1,
        parch=0,
        ticket="TEST123",
        fare=7.25,
        cabin=None,
        embarked="S",
        survived=None
    )

def test_model_training(model_service, sample_passenger):
    """Test model training functionality."""
    # Train model with sample data
    metrics = model_service.train([sample_passenger])
    
    # Verify metrics
    assert isinstance(metrics, ModelMetrics)
    assert 0 <= metrics.accuracy <= 1
    assert 0 <= metrics.precision <= 1
    assert 0 <= metrics.recall <= 1
    assert 0 <= metrics.f1_score <= 1
    assert metrics.training_timestamp is not None
    assert metrics.feature_importance is not None

def test_model_prediction(model_service, sample_passenger):
    """Test model prediction functionality."""
    # Train model first
    model_service.train([sample_passenger])
    
    # Make prediction
    prediction = model_service.predict(sample_passenger)
    
    # Verify prediction
    assert isinstance(prediction, ModelPrediction)
    assert prediction.passenger_id == sample_passenger.passenger_id
    assert 0 <= prediction.survival_probability <= 1
    assert isinstance(prediction.predicted_survival, bool)
    assert prediction.prediction_timestamp is not None
    assert prediction.model_version is not None
    assert prediction.feature_importance is not None

def test_prediction_without_training(model_service, sample_passenger):
    """Test prediction without training the model first."""
    with pytest.raises(ModelNotFoundError):
        model_service.predict(sample_passenger)

def test_model_evaluation(model_service, sample_passenger):
    """Test model evaluation functionality."""
    # Train model first
    model_service.train([sample_passenger])
    
    # Evaluate model
    metrics = model_service.evaluate([sample_passenger])
    
    # Verify metrics
    assert isinstance(metrics, ModelMetrics)
    assert 0 <= metrics.accuracy <= 1
    assert 0 <= metrics.precision <= 1
    assert 0 <= metrics.recall <= 1
    assert 0 <= metrics.f1_score <= 1
    assert metrics.training_timestamp is not None
    assert metrics.feature_importance is not None
