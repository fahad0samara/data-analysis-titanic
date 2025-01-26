"""Tests for data utility functions."""
import pytest
import pandas as pd
import numpy as np
from src.utils.data_utils import (
    load_and_preprocess_data,
    encode_categorical_features,
    create_age_groups,
    create_fare_groups,
    calculate_survival_statistics
)

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return pd.DataFrame({
        'PassengerId': range(1, 6),
        'Survived': [0, 1, 1, 0, 1],
        'Pclass': [1, 2, 3, 1, 2],
        'Name': ['John Doe', 'Jane Smith', 'Bob Brown', 'Alice White', 'Tom Gray'],
        'Sex': ['male', 'female', 'male', 'female', 'male'],
        'Age': [25, 30, np.nan, 45, 50],
        'SibSp': [1, 0, 2, 1, 0],
        'Parch': [0, 2, 1, 0, 1],
        'Ticket': ['123', '456', '789', '012', '345'],
        'Fare': [50, 30, np.nan, 100, 75],
        'Cabin': ['C123', np.nan, 'E45', 'B78', np.nan],
        'Embarked': ['S', 'C', np.nan, 'Q', 'S']
    })

def test_load_and_preprocess_data(sample_data, tmp_path):
    """Test data loading and preprocessing."""
    # Save sample data to temporary file
    data_path = tmp_path / "test_titanic.csv"
    sample_data.to_csv(data_path, index=False)
    
    # Load and preprocess data
    df = load_and_preprocess_data(str(data_path))
    
    # Check if missing values are handled
    assert df['Age'].isna().sum() == 0
    assert df['Fare'].isna().sum() == 0
    assert df['Embarked'].isna().sum() == 0
    
    # Check if new features are created
    assert 'FamilySize' in df.columns
    assert 'IsAlone' in df.columns
    assert 'Title' in df.columns

def test_encode_categorical_features(sample_data):
    """Test categorical feature encoding."""
    categorical_columns = ['Sex', 'Embarked']
    df_encoded, encoders = encode_categorical_features(sample_data, categorical_columns)
    
    # Check if encoders are created for all categorical columns
    assert set(encoders.keys()) == set(categorical_columns)
    
    # Check if encoded values are numeric
    for col in categorical_columns:
        assert df_encoded[col].dtype in ['int32', 'int64']

def test_create_age_groups(sample_data):
    """Test age group creation."""
    df = create_age_groups(sample_data)
    
    # Check if AgeGroup column is created
    assert 'AgeGroup' in df.columns
    
    # Check if all age groups are present
    expected_groups = ['Very Young', 'Young', 'Middle', 'Senior', 'Elderly']
    actual_groups = sorted(df['AgeGroup'].unique())
    assert all(group in expected_groups for group in actual_groups)

def test_create_fare_groups(sample_data):
    """Test fare group creation."""
    df = create_fare_groups(sample_data)
    
    # Check if FareGroup column is created
    assert 'FareGroup' in df.columns
    
    # Check if all fare groups are present
    expected_groups = ['Low', 'Medium', 'High', 'Very High']
    actual_groups = sorted(df['FareGroup'].unique())
    assert all(group in expected_groups for group in actual_groups)

def test_calculate_survival_statistics(sample_data):
    """Test survival statistics calculation."""
    stats = calculate_survival_statistics(sample_data)
    
    # Check if all required statistics are present
    assert 'overall_survival_rate' in stats
    assert 'survival_by_class' in stats
    assert 'survival_by_sex' in stats
    assert 'survival_by_family_size' in stats
    
    # Check if overall survival rate is calculated correctly
    expected_survival_rate = sample_data['Survived'].mean()
    assert stats['overall_survival_rate'] == expected_survival_rate
