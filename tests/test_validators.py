"""Tests for data validators."""
import pytest
from datetime import datetime
from src.domain.entities import Passenger, ModelPrediction
from src.domain.validators import validate_passenger, validate_prediction

def test_validate_valid_passenger():
    """Test validation of a valid passenger."""
    passenger = Passenger(
        passenger_id=1,
        pclass=1,
        sex=0,  # male
        age=30.0,
        sibsp=1,
        parch=0,
        fare=50.0,
        embarked=0  # S
    )
    assert validate_passenger(passenger) is None

def test_validate_invalid_passenger_id():
    """Test validation with invalid passenger ID."""
    passenger = Passenger(
        passenger_id=-1,  # Invalid ID
        pclass=1,
        sex=0,
        age=30.0,
        sibsp=1,
        parch=0,
        fare=50.0,
        embarked=0
    )
    with pytest.raises(ValueError):
        validate_passenger(passenger)

def test_validate_invalid_pclass():
    """Test validation with invalid passenger class."""
    passenger = Passenger(
        passenger_id=1,
        pclass=4,  # Invalid class (should be 1, 2, or 3)
        sex=0,
        age=30.0,
        sibsp=1,
        parch=0,
        fare=50.0,
        embarked=0
    )
    with pytest.raises(ValueError):
        validate_passenger(passenger)

def test_validate_invalid_gender():
    """Test validation with invalid gender."""
    passenger = Passenger(
        passenger_id=1,
        pclass=1,
        sex=2,  # Invalid gender (should be 0 or 1)
        age=30.0,
        sibsp=1,
        parch=0,
        fare=50.0,
        embarked=0
    )
    with pytest.raises(ValueError):
        validate_passenger(passenger)

def test_validate_invalid_age():
    """Test validation with invalid age."""
    passenger = Passenger(
        passenger_id=1,
        pclass=1,
        sex=0,
        age=-5.0,  # Invalid age
        sibsp=1,
        parch=0,
        fare=50.0,
        embarked=0
    )
    with pytest.raises(ValueError):
        validate_passenger(passenger)

def test_validate_valid_prediction():
    """Test validation of a valid prediction."""
    prediction = ModelPrediction(
        passenger_id=1,
        survival_probability=0.75,
        predicted_survival=True
    )
    assert validate_prediction(prediction) is None

def test_validate_invalid_probability():
    """Test validation with invalid probability."""
    prediction = ModelPrediction(
        passenger_id=1,
        survival_probability=1.5,  # Invalid probability (should be between 0 and 1)
        predicted_survival=True
    )
    with pytest.raises(ValueError):
        validate_prediction(prediction)
