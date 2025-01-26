"""Utility functions for data processing and manipulation."""
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data(filepath: str) -> pd.DataFrame:
    """Load and preprocess the Titanic dataset."""
    df = pd.read_csv(filepath)
    
    # Convert categorical variables
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    
    # Fill missing values
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    
    # Create family size feature
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    
    # Create new features
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    
    # Extract title from name using raw string
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
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
    # Ensure Age is numeric
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    
    bins = [0, 12, 18, 35, 50, 65, float('inf')]
    labels = ['Child', 'Teen', 'Young Adult', 'Adult', 'Senior', 'Elderly']
    df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
    return df

def create_fare_groups(df: pd.DataFrame, num_bins: int = 4) -> pd.DataFrame:
    """Create fare groups from continuous fare values."""
    df = df.copy()
    # Ensure Fare is numeric
    df['Fare'] = pd.to_numeric(df['Fare'], errors='coerce')
    
    bins = [0, 20, 50, 100, 200, float('inf')]
    labels = ['Low', 'Medium-Low', 'Medium', 'Medium-High', 'High']
    df['FareGroup'] = pd.cut(df['Fare'], bins=bins, labels=labels, right=False)
    return df

def calculate_survival_statistics(df: pd.DataFrame) -> Dict:
    """Calculate various survival statistics from the dataset."""
    stats = {}
    
    # Ensure FamilySize exists
    if 'FamilySize' not in df.columns:
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    
    # Overall survival rate
    stats['overall_survival_rate'] = df['Survived'].mean()
    
    # Survival rate by passenger class
    stats['survival_by_class'] = df.groupby('Pclass')['Survived'].mean().to_dict()
    
    # Survival rate by sex
    stats['survival_by_sex'] = df.groupby('Sex')['Survived'].mean().to_dict()
    
    # Survival rate by family size
    stats['survival_by_family_size'] = df.groupby('FamilySize')['Survived'].mean().to_dict()
    
    # Survival rate by age group
    if 'AgeGroup' in df.columns:
        stats['survival_by_age_group'] = df.groupby('AgeGroup')['Survived'].mean().to_dict()
    
    # Survival rate by fare group
    if 'FareGroup' in df.columns:
        stats['survival_by_fare_group'] = df.groupby('FareGroup')['Survived'].mean().to_dict()
    
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

def get_feature_columns() -> List[str]:
    """Return the list of feature columns used for model training."""
    return ['Age', 'Fare', 'SibSp', 'Parch', 'Sex', 'Embarked', 'Pclass']
