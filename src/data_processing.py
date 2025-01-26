import pandas as pd
import numpy as np
from typing import Optional, Union, List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self):
        """Initialize the DataProcessor class"""
        self.data = None
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load data from a file
        
        Args:
            filepath (str): Path to the data file
            
        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            self.data = pd.read_csv(filepath)
            logger.info(f"Successfully loaded data from {filepath}")
            return self.data
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
            
    def clean_data(self, 
                   columns: Optional[List[str]] = None,
                   drop_duplicates: bool = True,
                   fill_na: Union[str, float, None] = None) -> pd.DataFrame:
        """
        Clean the input dataframe
        
        Args:
            columns (List[str], optional): Specific columns to clean
            drop_duplicates (bool): Whether to remove duplicate rows
            fill_na (str, float, optional): Value to fill NaN with
            
        Returns:
            pd.DataFrame: Cleaned dataframe
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first using load_data()")
            
        df = self.data.copy()
        
        try:
            # Select specific columns if provided
            if columns:
                df = df[columns]
            
            # Remove duplicates if requested
            if drop_duplicates:
                df = df.drop_duplicates()
                
            # Handle missing values
            if fill_na is not None:
                df = df.fillna(fill_na)
            else:
                df = df.dropna()
                
            logger.info("Data cleaning completed successfully")
            return df
            
        except Exception as e:
            logger.error(f"Error cleaning data: {str(e)}")
            raise
