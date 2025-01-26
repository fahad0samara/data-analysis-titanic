"""Tests for data utility functions."""
import pandas as pd
import pytest
from src.utils.data_utils import (
    create_age_groups,
    create_fare_groups,
    calculate_survival_statistics,
    get_feature_columns
)

@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'Age': [25, 35, 45, 55, 65],
        'Fare': [10.0, 30.0, 60.0, 120.0, 250.0],
        'Survived': [0, 1, 1, 0, 1],
        'Pclass': [1, 2, 3, 1, 2],
        'Sex': [0, 1, 0, 1, 0],  # 0: male, 1: female
        'SibSp': [1, 0, 2, 1, 0],
        'Parch': [0, 2, 1, 0, 1],
        'Embarked': [0, 1, 2, 0, 1]  # 0: S, 1: C, 2: Q
    })

def test_create_age_groups():
    """Test age group creation."""
    df = pd.DataFrame({
        'Age': [5, 15, 25, 45, 70]
    })
    result = create_age_groups(df)
    assert 'AgeGroup' in result.columns
    assert len(result['AgeGroup'].unique()) > 0
    assert not result['AgeGroup'].isna().any()

def test_create_fare_groups():
    """Test fare group creation."""
    df = pd.DataFrame({
        'Fare': [10.0, 30.0, 60.0, 120.0, 250.0]
    })
    result = create_fare_groups(df)
    assert 'FareGroup' in result.columns
    assert len(result['FareGroup'].unique()) > 0
    assert not result['FareGroup'].isna().any()

def test_calculate_survival_statistics(sample_df):
    """Test survival statistics calculation."""
    stats = calculate_survival_statistics(sample_df)
    assert 'overall_survival_rate' in stats
    assert 'survival_by_class' in stats
    assert 'survival_by_sex' in stats
    assert isinstance(stats['overall_survival_rate'], float)
    assert isinstance(stats['survival_by_class'], dict)
    assert isinstance(stats['survival_by_sex'], dict)

def test_get_feature_columns():
    """Test getting feature columns."""
    features = get_feature_columns()
    assert isinstance(features, list)
    assert len(features) > 0
    assert 'Age' in features
    assert 'Sex' in features
    assert 'Pclass' in features
