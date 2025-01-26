import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, List, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineering:
    def __init__(self):
        """Initialize the FeatureEngineering class"""
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def create_features(self, 
                       df: pd.DataFrame,
                       numeric_columns: Optional[List[str]] = None,
                       categorical_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Create and transform features from the input dataframe
        
        Args:
            df (pd.DataFrame): Input dataframe
            numeric_columns (List[str], optional): Columns to scale
            categorical_columns (List[str], optional): Columns to encode
            
        Returns:
            pd.DataFrame: Transformed dataframe with new features
        """
        try:
            df_transformed = df.copy()
            
            # Scale numeric features
            if numeric_columns:
                df_transformed[numeric_columns] = self.scaler.fit_transform(df[numeric_columns])
                logger.info("Numeric features scaled successfully")
            
            # Encode categorical features
            if categorical_columns:
                for col in categorical_columns:
                    le = LabelEncoder()
                    df_transformed[col] = le.fit_transform(df[col])
                    self.label_encoders[col] = le
                logger.info("Categorical features encoded successfully")
            
            return df_transformed
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {str(e)}")
            raise
            
    def transform_features(self, 
                          df: pd.DataFrame,
                          numeric_columns: Optional[List[str]] = None,
                          categorical_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Transform new data using fitted scalers and encoders
        
        Args:
            df (pd.DataFrame): Input dataframe
            numeric_columns (List[str], optional): Columns to scale
            categorical_columns (List[str], optional): Columns to encode
            
        Returns:
            pd.DataFrame: Transformed dataframe
        """
        try:
            df_transformed = df.copy()
            
            # Transform numeric features
            if numeric_columns:
                df_transformed[numeric_columns] = self.scaler.transform(df[numeric_columns])
            
            # Transform categorical features
            if categorical_columns:
                for col in categorical_columns:
                    if col in self.label_encoders:
                        df_transformed[col] = self.label_encoders[col].transform(df[col])
                    else:
                        raise ValueError(f"No encoder found for column {col}")
            
            return df_transformed
            
        except Exception as e:
            logger.error(f"Error in feature transformation: {str(e)}")
            raise
