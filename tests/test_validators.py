"""Tests for data validators."""
import pytest
from datetime import datetime
from src.domain.entities import Passenger, ModelPrediction, ModelMetrics
from src.infrastructure.validators import PassengerValidator, DatasetValidator

@pytest.fixture
def validator():
    """Create a validator instance for testing."""
    return PassengerValidator()

@pytest.fixture
def valid_passenger():
    """Create a valid passenger instance for testing."""
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

@pytest.fixture
def valid_prediction():
    """Create a valid prediction instance for testing."""
    return ModelPrediction(
        passenger_id=1,
        survival_probability=0.75,
        predicted_survival=True,
        prediction_timestamp=datetime.now(),
        model_version="test_v1",
        feature_importance={"age": 0.5, "fare": 0.5}
    )

@pytest.fixture
def valid_metrics():
    """Create valid model metrics for testing."""
    return ModelMetrics(
        model_version="test_v1",
        accuracy=0.85,
        precision=0.83,
        recall=0.87,
        f1_score=0.85,
        roc_auc=0.89,
        training_timestamp=datetime.now(),
        feature_importance={"age": 0.5, "fare": 0.5},
        confusion_matrix=[[100, 20], [15, 85]]
    )

def test_validate_valid_passenger(validator, valid_passenger):
    """Test validation of a valid passenger."""
    assert validator.validate_passenger(valid_passenger) is True

def test_validate_invalid_passenger_id(validator, valid_passenger):
    """Test validation of invalid passenger ID."""
    valid_passenger.passenger_id = -1
    assert validator.validate_passenger(valid_passenger) is False

def test_validate_invalid_pclass(validator, valid_passenger):
    """Test validation of invalid passenger class."""
    valid_passenger.pclass = 4
    assert validator.validate_passenger(valid_passenger) is False

def test_validate_invalid_gender(validator, valid_passenger):
    """Test validation of invalid gender."""
    valid_passenger.sex = "invalid"
    assert validator.validate_passenger(valid_passenger) is False

def test_validate_invalid_age(validator, valid_passenger):
    """Test validation of invalid age."""
    valid_passenger.age = -1
    assert validator.validate_passenger(valid_passenger) is False

def test_validate_valid_prediction(validator, valid_prediction):
    """Test validation of a valid prediction."""
    assert validator.validate_prediction(valid_prediction) is True

def test_validate_invalid_probability(validator, valid_prediction):
    """Test validation of invalid probability."""
    valid_prediction.survival_probability = 1.5
    assert validator.validate_prediction(valid_prediction) is False

def test_validate_valid_metrics(validator, valid_metrics):
    """Test validation of valid metrics."""
    assert validator.validate_metrics(valid_metrics) is True

def test_validate_invalid_metrics_values(validator, valid_metrics):
    """Test validation of invalid metric values."""
    valid_metrics.accuracy = 1.5
    assert validator.validate_metrics(valid_metrics) is False

def test_dataset_validator_schema():
    """Test dataset schema validation."""
    validator = DatasetValidator()
    data = {"field1": "value1", "field2": "value2"}
    required_fields = {"field1", "field2"}
    assert validator.validate_schema(data, required_fields) is True

def test_dataset_validator_data_types():
    """Test dataset data type validation."""
    validator = DatasetValidator()
    data = {"string_field": "text", "int_field": 42}
    type_mapping = {"string_field": str, "int_field": int}
    assert validator.validate_data_types(data, type_mapping) is True

def test_dataset_validator_value_ranges():
    """Test dataset value range validation."""
    validator = DatasetValidator()
    data = {"age": 25, "score": 85}
    range_mapping = {"age": (0, 120), "score": (0, 100)}
    assert validator.validate_value_ranges(data, range_mapping) is True

def test_dataset_validator_categorical_values():
    """Test dataset categorical value validation."""
    validator = DatasetValidator()
    data = {"gender": "male", "class": "first"}
    categorical_mapping = {
        "gender": {"male", "female"},
        "class": {"first", "second", "third"}
    }
    assert validator.validate_categorical_values(data, categorical_mapping) is True
