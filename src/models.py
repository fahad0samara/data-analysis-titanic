import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from typing import Tuple, Any, Dict
import joblib
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, model: Any):
        """
        Initialize the ModelTrainer class
        
        Args:
            model: A scikit-learn compatible model
        """
        self.model = model
        
    def split_data(self, 
                   X: pd.DataFrame,
                   y: pd.Series,
                   test_size: float = 0.2,
                   random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into training and testing sets
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target variable
            test_size (float): Proportion of dataset to include in the test split
            random_state (int): Random state for reproducibility
            
        Returns:
            Tuple containing train-test split of inputs
        """
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            logger.info("Data split successfully")
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logger.error(f"Error splitting data: {str(e)}")
            raise
            
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Train the model
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
        """
        try:
            self.model.fit(X_train, y_train)
            logger.info("Model trained successfully")
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise
            
    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        Evaluate the model performance
        
        Args:
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
            
        Returns:
            Dict: Dictionary containing evaluation metrics
        """
        try:
            y_pred = self.model.predict(X_test)
            
            # Calculate metrics based on problem type (classification or regression)
            if len(np.unique(y_test)) <= 10:  # Classification
                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'classification_report': classification_report(y_test, y_pred)
                }
            else:  # Regression
                metrics = {
                    'mse': mean_squared_error(y_test, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
                }
                
            logger.info("Model evaluation completed")
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            raise
            
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to disk
        
        Args:
            filepath (str): Path where to save the model
        """
        try:
            joblib.dump(self.model, filepath)
            logger.info(f"Model saved successfully to {filepath}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
            
    @staticmethod
    def load_model(filepath: str) -> Any:
        """
        Load a trained model from disk
        
        Args:
            filepath (str): Path to the saved model
            
        Returns:
            Loaded model
        """
        try:
            model = joblib.load(filepath)
            logger.info(f"Model loaded successfully from {filepath}")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
