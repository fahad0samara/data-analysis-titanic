"""Configuration settings for the Titanic Survival Analysis project."""
import os
from pathlib import Path

# Project root directory
ROOT_DIR = Path(__file__).parent.parent.parent

# Data directories
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
INTERIM_DATA_DIR = DATA_DIR / "interim"

# Model directories
MODELS_DIR = ROOT_DIR / "models"
MODEL_ARTIFACTS_DIR = MODELS_DIR / "artifacts"
MODEL_METRICS_DIR = MODELS_DIR / "metrics"

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, INTERIM_DATA_DIR,
                 MODEL_ARTIFACTS_DIR, MODEL_METRICS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Data file paths
TITANIC_RAW_DATA = RAW_DATA_DIR / "titanic.csv"
TITANIC_PROCESSED_DATA = PROCESSED_DATA_DIR / "titanic_processed.csv"

# Model file paths
BEST_MODEL_PATH = MODEL_ARTIFACTS_DIR / "best_model.joblib"
MODEL_METRICS_PATH = MODEL_METRICS_DIR / "model_metrics.json"

# Model parameters
MODEL_PARAMS = {
    'n_estimators': 100,
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'random_state': 42
}

# Feature engineering parameters
FEATURE_ENGINEERING_PARAMS = {
    'age_bins': 10,
    'fare_bins': 10,
    'categorical_features': ['Sex', 'Embarked'],
    'numerical_features': ['Age', 'Fare', 'SibSp', 'Parch'],
    'target': 'Survived'
}

# API settings
API_HOST = "0.0.0.0"
API_PORT = 8000
API_WORKERS = 4

# Dashboard settings
DASHBOARD_PORT = 8501
DASHBOARD_THEME = "light"
