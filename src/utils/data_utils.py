"""Utility functions for data processing and manipulation."""
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data(filepath: str) -> pd.DataFrame:
    """Load and preprocess the Titanic dataset."""
    df = pd.read_csv(filepath)
    
    # Handle missing values
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    
    # Create new features
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    
    # Extract title from name
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 
                                     'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    
    return df

def encode_categorical_features(df: pd.DataFrame, categorical_columns: List[str]) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """Encode categorical features using LabelEncoder."""
    df_encoded = df.copy()
    encoders = {}
    
    for column in categorical_columns:
        if column in df.columns:
            encoder = LabelEncoder()
            df_encoded[column] = encoder.fit_transform(df[column].astype(str))
            encoders[column] = encoder
    
    return df_encoded, encoders

def create_age_groups(df: pd.DataFrame, num_bins: int = 5) -> pd.DataFrame:
    """Create age groups from continuous age values."""
    df = df.copy()
    df['AgeGroup'] = pd.qcut(df['Age'], q=num_bins, labels=['Very Young', 'Young', 'Middle', 'Senior', 'Elderly'])
    return df

def create_fare_groups(df: pd.DataFrame, num_bins: int = 4) -> pd.DataFrame:
    """Create fare groups from continuous fare values."""
    df = df.copy()
    df['FareGroup'] = pd.qcut(df['Fare'], q=num_bins, labels=['Low', 'Medium', 'High', 'Very High'])
    return df

def calculate_survival_statistics(df: pd.DataFrame) -> Dict:
    """Calculate various survival statistics from the dataset."""
    stats = {
        'overall_survival_rate': df['Survived'].mean(),
        'survival_by_class': df.groupby('Pclass')['Survived'].mean().to_dict(),
        'survival_by_sex': df.groupby('Sex')['Survived'].mean().to_dict(),
        'survival_by_age_group': df.groupby('AgeGroup')['Survived'].mean().to_dict() if 'AgeGroup' in df.columns else None,
        'survival_by_fare_group': df.groupby('FareGroup')['Survived'].mean().to_dict() if 'FareGroup' in df.columns else None,
        'survival_by_family_size': df.groupby('FamilySize')['Survived'].mean().to_dict()
    }
    return stats

def generate_feature_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Generate interaction features between important variables."""
    df = df.copy()
    
    # Pclass and Sex interaction
    df['Pclass_Sex'] = df['Pclass'].astype(str) + '_' + df['Sex']
    
    # Age and Class interaction
    df['Age_Class'] = df['Age'] * df['Pclass']
    
    # Fare and Class interaction
    df['Fare_Per_Class'] = df['Fare'] / df['Pclass']
    
    # Family size and Pclass interaction
    df['Family_Class'] = df['FamilySize'] * df['Pclass']
    
    return df

def prepare_features_for_model(df: pd.DataFrame, feature_columns: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare features and target variable for model training."""
    X = df[feature_columns].copy()
    y = df['Survived'] if 'Survived' in df.columns else None
    
    # Handle any remaining missing values
    X = X.fillna(X.mean())
    
    return X, y

def calculate_correlation_matrix(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """Calculate correlation matrix for specified features."""
    return df[features].corr()
