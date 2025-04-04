"""
Dataset Handler Module for Julius AI Chatbot.

This module provides functionality to load, process, and manage datasets
from various file formats (CSV, Excel, etc.).
"""

import os
import pandas as pd
import numpy as np
import json
from typing import Dict, List, Optional, Any, Tuple
import streamlit as st


class DatasetHandler:
    """
    A class to handle dataset loading and processing.
    
    This class provides methods to:
    - Load datasets from various file formats
    - Process and clean datasets
    - Convert datasets to different formats
    - Save datasets to disk
    """
    
    def __init__(self, temp_dir: str = "temp_files"):
        """
        Initialize the Dataset Handler.
        
        Args:
            temp_dir: Directory for temporary files
        """
        self.temp_dir = temp_dir
        self.current_dataset = None
        self.current_dataset_name = None
        self.current_dataset_source = None
        
        # Create temp directory if it doesn't exist
        os.makedirs(self.temp_dir, exist_ok=True)
    
    def load_dataset_from_file(self, uploaded_file) -> bool:
        """
        Load a dataset from an uploaded file.
        
        Args:
            uploaded_file: Uploaded file from Streamlit
            
        Returns:
            bool: True if loading successful, False otherwise
        """
        try:
            # Get file extension
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            # Load dataset based on file extension
            if file_extension == 'csv':
                df = pd.read_csv(uploaded_file)
            elif file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(uploaded_file)
            elif file_extension == 'json':
                df = pd.read_json(uploaded_file)
            elif file_extension == 'parquet':
                df = pd.read_parquet(uploaded_file)
            elif file_extension == 'feather':
                df = pd.read_feather(uploaded_file)
            elif file_extension == 'pickle' or file_extension == 'pkl':
                df = pd.read_pickle(uploaded_file)
            else:
                st.error(f"Unsupported file format: {file_extension}")
                return False
            
            # Store dataset information
            self.current_dataset = df
            self.current_dataset_name = uploaded_file.name
            self.current_dataset_source = 'upload'
            
            # Save dataset to temp directory
            self._save_dataset_to_temp()
            
            return True
            
        except Exception as e:
            st.error(f"Error loading dataset: {str(e)}")
            return False
    
    def load_dataset_from_sql(self, df: pd.DataFrame, name: str) -> bool:
        """
        Load a dataset from SQL query results.
        
        Args:
            df: DataFrame from SQL query
            name: Name for the dataset
            
        Returns:
            bool: True if loading successful, False otherwise
        """
        try:
            # Store dataset information
            self.current_dataset = df
            self.current_dataset_name = name
            self.current_dataset_source = 'sql'
            
            # Save dataset to temp directory
            self._save_dataset_to_temp()
            
            return True
            
        except Exception as e:
            st.error(f"Error loading dataset from SQL: {str(e)}")
            return False
    
    def load_dataset_from_pdf(self, df: pd.DataFrame, name: str) -> bool:
        """
        Load a dataset from PDF content.
        
        Args:
            df: DataFrame from PDF content
            name: Name for the dataset
            
        Returns:
            bool: True if loading successful, False otherwise
        """
        try:
            # Store dataset information
            self.current_dataset = df
            self.current_dataset_name = name
            self.current_dataset_source = 'pdf'
            
            # Save dataset to temp directory
            self._save_dataset_to_temp()
            
            return True
            
        except Exception as e:
            st.error(f"Error loading dataset from PDF: {str(e)}")
            return False
    
    def load_dataset_from_dataframe(self, df: pd.DataFrame, name: str) -> bool:
        """
        Load a dataset directly from a DataFrame.
        
        Args:
            df: DataFrame to load
            name: Name for the dataset
            
        Returns:
            bool: True if loading successful, False otherwise
        """
        try:
            # Store dataset information
            self.current_dataset = df
            self.current_dataset_name = name
            self.current_dataset_source = 'dataframe'
            
            # Save dataset to temp directory
            self._save_dataset_to_temp()
            
            return True
            
        except Exception as e:
            st.error(f"Error loading dataset from DataFrame: {str(e)}")
            return False
    
    def _save_dataset_to_temp(self) -> None:
        """
        Save the current dataset to the temp directory.
        """
        if self.current_dataset is None or self.current_dataset_name is None:
            return
            
        try:
            # Save with original name
            original_path = os.path.join(self.temp_dir, self.current_dataset_name)
            self.current_dataset.to_csv(original_path, index=False)
            
            # Also save with a fixed name for consistency
            fixed_path = os.path.join(self.temp_dir, "current_dataset.csv")
            self.current_dataset.to_csv(fixed_path, index=False)
            
        except Exception as e:
            st.error(f"Error saving dataset to temp directory: {str(e)}")
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get information about the current dataset.
        
        Returns:
            Dict[str, Any]: Dataset information
        """
        if self.current_dataset is None:
            return {}
            
        try:
            # Get basic dataset information
            info = {
                'name': self.current_dataset_name,
                'source': self.current_dataset_source,
                'shape': self.current_dataset.shape,
                'columns': self.current_dataset.columns.tolist(),
                'dtypes': {col: str(dtype) for col, dtype in self.current_dataset.dtypes.items()},
                'memory_usage': self.current_dataset.memory_usage(deep=True).sum(),
                'has_missing_values': self.current_dataset.isna().any().any(),
                'missing_values_count': self.current_dataset.isna().sum().sum()
            }
            
            return info
            
        except Exception as e:
            st.error(f"Error getting dataset info: {str(e)}")
            return {}
    
    def get_dataset_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current dataset.
        
        Returns:
            Dict[str, Any]: Dataset summary
        """
        if self.current_dataset is None:
            return {}
            
        try:
            # Get dataset summary
            summary = {
                'name': self.current_dataset_name,
                'source': self.current_dataset_source,
                'shape': self.current_dataset.shape,
                'columns': self.current_dataset.columns.tolist(),
                'numeric_columns': self.current_dataset.select_dtypes(include=[np.number]).columns.tolist(),
                'categorical_columns': self.current_dataset.select_dtypes(include=['object', 'category']).columns.tolist(),
                'datetime_columns': self.current_dataset.select_dtypes(include=['datetime']).columns.tolist(),
                'missing_values': {col: int(self.current_dataset[col].isna().sum()) for col in self.current_dataset.columns},
                'numeric_summary': {},
                'categorical_summary': {}
            }
            
            # Add numeric summary
            for col in summary['numeric_columns']:
                summary['numeric_summary'][col] = {
                    'min': float(self.current_dataset[col].min()) if not pd.isna(self.current_dataset[col].min()) else None,
                    'max': float(self.current_dataset[col].max()) if not pd.isna(self.current_dataset[col].max()) else None,
                    'mean': float(self.current_dataset[col].mean()) if not pd.isna(self.current_dataset[col].mean()) else None,
                    'median': float(self.current_dataset[col].median()) if not pd.isna(self.current_dataset[col].median()) else None,
                    'std': float(self.current_dataset[col].std()) if not pd.isna(self.current_dataset[col].std()) else None
                }
            
            # Add categorical summary
            for col in summary['categorical_columns']:
                value_counts = self.current_dataset[col].value_counts().head(10).to_dict()
                summary['categorical_summary'][col] = {
                    'unique_values': int(self.current_dataset[col].nunique()),
                    'top_values': {str(k): int(v) for k, v in value_counts.items()}
                }
            
            return summary
            
        except Exception as e:
            st.error(f"Error getting dataset summary: {str(e)}")
            return {}
    
    def clean_dataset(self) -> bool:
        """
        Clean the current dataset.
        
        Returns:
            bool: True if cleaning successful, False otherwise
        """
        if self.current_dataset is None:
            return False
            
        try:
            # Make a copy of the dataset
            df = self.current_dataset.copy()
            
            # Convert column names to snake_case
            df.columns = [self._to_snake_case(col) for col in df.columns]
            
            # Handle missing values
            for col in df.columns:
                # If numeric column, fill with median
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].median())
                # If categorical column, fill with mode
                elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")
            
            # Try to convert string columns to datetime
            for col in df.select_dtypes(include=['object']).columns:
                try:
                    # Check if column contains date-like strings
                    if df[col].str.contains(r'\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|\d{2}-\d{2}-\d{4}', regex=True).any():
                        df[col] = pd.to_datetime(df[col], errors='ignore')
                except Exception:
                    pass
            
            # Update the current dataset
            self.current_dataset = df
            
            # Save the cleaned dataset
            self._save_dataset_to_temp()
            
            return True
            
        except Exception as e:
            st.error(f"Error cleaning dataset: {str(e)}")
            return False
    
    def _to_snake_case(self, name: str) -> str:
        """
        Convert a string to snake_case.
        
        Args:
            name: String to convert
            
        Returns:
            str: snake_case string
        """
        import re
        # Replace spaces and special characters with underscores
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        s2 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1)
        # Replace non-alphanumeric characters with underscores
        s3 = re.sub(r'[^a-zA-Z0-9]', '_', s2)
        # Convert to lowercase and remove duplicate underscores
        s4 = re.sub(r'_+', '_', s3).lower().strip('_')
        return s4
    
    def get_dataset_for_julius(self) -> Tuple[pd.DataFrame, str, Dict[str, Any]]:
        """
        Get the current dataset in a format suitable for Julius AI.
        
        Returns:
            Tuple containing:
            - pd.DataFrame: Dataset
            - str: Dataset name
            - Dict[str, Any]: Dataset information
        """
        if self.current_dataset is None:
            return None, None, {}
            
        # Get dataset information
        info = self.get_dataset_info()
        
        # Return dataset, name, and info
        return self.current_dataset, self.current_dataset_name, info
    
    def export_dataset(self, format: str = 'csv') -> Optional[str]:
        """
        Export the current dataset to a file.
        
        Args:
            format: Export format ('csv', 'excel', 'json', etc.)
            
        Returns:
            Optional[str]: Path to exported file or None if export failed
        """
        if self.current_dataset is None:
            return None
            
        try:
            # Create export filename
            base_name = os.path.splitext(self.current_dataset_name)[0]
            
            # Export based on format
            if format.lower() == 'csv':
                export_path = f"{base_name}_export.csv"
                self.current_dataset.to_csv(export_path, index=False)
            elif format.lower() == 'excel':
                export_path = f"{base_name}_export.xlsx"
                self.current_dataset.to_excel(export_path, index=False)
            elif format.lower() == 'json':
                export_path = f"{base_name}_export.json"
                self.current_dataset.to_json(export_path, orient='records')
            elif format.lower() == 'parquet':
                export_path = f"{base_name}_export.parquet"
                self.current_dataset.to_parquet(export_path, index=False)
            else:
                st.error(f"Unsupported export format: {format}")
                return None
            
            return export_path
            
        except Exception as e:
            st.error(f"Error exporting dataset: {str(e)}")
            return None
    
    def clear_dataset(self) -> None:
        """
        Clear the current dataset.
        """
        self.current_dataset = None
        self.current_dataset_name = None
        self.current_dataset_source = None