"""
Data Service Module for Multi-Source Julius AI Chatbot.

This module provides the DataService class for handling data loading, processing, and storage.
It uses Polars for faster data processing and provides methods for loading data from various sources.
"""

import os
import io
import tempfile
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path

import polars as pl
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field

class DataStats(BaseModel):
    """Data statistics model."""
    row_count: int = Field(..., description="Number of rows in the dataset")
    column_count: int = Field(..., description="Number of columns in the dataset")
    memory_usage: str = Field(..., description="Memory usage of the dataset")
    column_types: Dict[str, str] = Field(..., description="Column types")
    missing_values: Dict[str, int] = Field(..., description="Missing values per column")
    numeric_stats: Optional[Dict[str, Dict[str, float]]] = Field(None, description="Statistics for numeric columns")

class DataService:
    """
    Service for handling data loading, processing, and storage.
    
    This service provides methods for:
    - Loading data from various sources (CSV, Excel, Parquet, JSON, etc.)
    - Processing data using Polars for faster operations
    - Calculating statistics and summaries
    - Exporting data to various formats
    """
    
    def __init__(self, cache_dir: str = "temp_files"):
        """
        Initialize the DataService.
        
        Args:
            cache_dir: Directory for caching data files
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self._current_dataset = None
        self._current_dataset_name = None
        self._current_dataset_source = None
    
    @property
    def current_dataset(self) -> Optional[pl.DataFrame]:
        """Get the current dataset."""
        return self._current_dataset
    
    @property
    def current_dataset_name(self) -> Optional[str]:
        """Get the current dataset name."""
        return self._current_dataset_name
    
    @property
    def current_dataset_source(self) -> Optional[str]:
        """Get the current dataset source."""
        return self._current_dataset_source
    
    def load_file(self, file, file_name: Optional[str] = None) -> Tuple[bool, Optional[pl.DataFrame], Optional[str]]:
        """
        Load data from a file.
        
        Args:
            file: File object or path
            file_name: Optional file name
            
        Returns:
            Tuple containing:
            - bool: Success status
            - Optional[pl.DataFrame]: Loaded dataframe or None
            - Optional[str]: Error message or None
        """
        try:
            # Get file extension
            if file_name is None:
                if hasattr(file, 'name'):
                    file_name = file.name
                else:
                    file_name = str(file)
            
            file_ext = Path(file_name).suffix.lower()
            
            # Create a temporary file if needed
            if hasattr(file, 'read'):
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
                    tmp.write(file.read())
                    tmp_path = tmp.name
                file_path = tmp_path
            else:
                file_path = file
            
            # Load data based on file extension
            if file_ext in ['.csv', '.txt']:
                df = pl.read_csv(file_path, infer_schema_length=10000)
            elif file_ext in ['.xlsx', '.xls']:
                # Polars doesn't support Excel directly, use pandas as bridge
                pandas_df = pd.read_excel(file_path)
                df = pl.from_pandas(pandas_df)
            elif file_ext == '.parquet':
                df = pl.read_parquet(file_path)
            elif file_ext == '.json':
                df = pl.read_json(file_path)
            elif file_ext == '.arrow':
                df = pl.read_ipc(file_path)
            elif file_ext == '.avro':
                df = pl.read_avro(file_path)
            else:
                return False, None, f"Unsupported file format: {file_ext}"
            
            # Set current dataset
            self._current_dataset = df
            self._current_dataset_name = Path(file_name).name
            self._current_dataset_source = 'file'
            
            # Save to cache
            cache_path = os.path.join(self.cache_dir, "current_dataset.parquet")
            df.write_parquet(cache_path)
            
            return True, df, None
        except Exception as e:
            return False, None, str(e)
    
    def load_from_dataframe(self, df: Union[pd.DataFrame, pl.DataFrame], name: str) -> Tuple[bool, Optional[pl.DataFrame], Optional[str]]:
        """
        Load data from a pandas or polars DataFrame.
        
        Args:
            df: Pandas or Polars DataFrame
            name: Dataset name
            
        Returns:
            Tuple containing:
            - bool: Success status
            - Optional[pl.DataFrame]: Loaded dataframe or None
            - Optional[str]: Error message or None
        """
        try:
            # Convert pandas DataFrame to polars if needed
            if isinstance(df, pd.DataFrame):
                pl_df = pl.from_pandas(df)
            else:
                pl_df = df
            
            # Set current dataset
            self._current_dataset = pl_df
            self._current_dataset_name = name
            self._current_dataset_source = 'dataframe'
            
            # Save to cache
            cache_path = os.path.join(self.cache_dir, "current_dataset.parquet")
            pl_df.write_parquet(cache_path)
            
            return True, pl_df, None
        except Exception as e:
            return False, None, str(e)
    
    def get_dataset_stats(self, df: Optional[pl.DataFrame] = None) -> DataStats:
        """
        Get statistics for a dataset.
        
        Args:
            df: Polars DataFrame (uses current dataset if None)
            
        Returns:
            DataStats: Dataset statistics
        """
        if df is None:
            df = self._current_dataset
        
        if df is None:
            raise ValueError("No dataset available")
        
        # Basic stats
        row_count = df.height
        column_count = df.width
        memory_usage = f"{df.estimated_size() / (1024 * 1024):.2f} MB"
        
        # Column types
        column_types = {col: str(df[col].dtype) for col in df.columns}
        
        # Missing values
        missing_values = {col: df[col].null_count() for col in df.columns}
        
        # Numeric stats
        numeric_stats = {}
        for col in df.columns:
            # Check if column is numeric using polars types
            if df[col].dtype in [pl.Float64, pl.Int64, pl.Float32, pl.Int32]:
                try:
                    stats = df[col].describe().to_dict()
                    numeric_stats[col] = {
                        'mean': stats.get('mean', 0.0),
                        'std': stats.get('std', 0.0),
                        'min': stats.get('min', 0.0),
                        'max': stats.get('max', 0.0),
                        'median': df[col].median()
                    }
                except:
                    pass
        
        return DataStats(
            row_count=row_count,
            column_count=column_count,
            memory_usage=memory_usage,
            column_types=column_types,
            missing_values=missing_values,
            numeric_stats=numeric_stats
        )
    
    def export_data(self, df: Optional[pl.DataFrame] = None, format: str = 'csv') -> bytes:
        """
        Export data to various formats.
        
        Args:
            df: Polars DataFrame (uses current dataset if None)
            format: Export format ('csv', 'excel', 'parquet', 'json')
            
        Returns:
            bytes: Exported data as bytes
        """
        if df is None:
            df = self._current_dataset
        
        if df is None:
            raise ValueError("No dataset available")
        
        buffer = io.BytesIO()
        
        if format.lower() == 'csv':
            csv_data = df.write_csv()
            buffer.write(csv_data.encode('utf-8'))
        elif format.lower() == 'excel':
            # Convert to pandas for Excel export
            pandas_df = df.to_pandas()
            pandas_df.to_excel(buffer, index=False, engine='openpyxl')
        elif format.lower() == 'parquet':
            df.write_parquet(buffer)
        elif format.lower() == 'json':
            json_data = df.write_json()
            buffer.write(json_data.encode('utf-8'))
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        buffer.seek(0)
        return buffer.getvalue()
    
    def process_data(self, operations: List[Dict[str, Any]], df: Optional[pl.DataFrame] = None) -> pl.DataFrame:
        """
        Process data using a list of operations.
        
        Args:
            operations: List of operations to apply
            df: Polars DataFrame (uses current dataset if None)
            
        Returns:
            pl.DataFrame: Processed dataframe
        """
        if df is None:
            df = self._current_dataset
        
        if df is None:
            raise ValueError("No dataset available")
        
        result = df
        
        for op in operations:
            op_type = op.get('type')
            
            if op_type == 'filter':
                column = op.get('column')
                operator = op.get('operator')
                value = op.get('value')
                
                if operator == '==':
                    result = result.filter(pl.col(column) == value)
                elif operator == '!=':
                    result = result.filter(pl.col(column) != value)
                elif operator == '>':
                    result = result.filter(pl.col(column) > value)
                elif operator == '>=':
                    result = result.filter(pl.col(column) >= value)
                elif operator == '<':
                    result = result.filter(pl.col(column) < value)
                elif operator == '<=':
                    result = result.filter(pl.col(column) <= value)
                elif operator == 'contains':
                    result = result.filter(pl.col(column).str.contains(value))
                elif operator == 'in':
                    result = result.filter(pl.col(column).is_in(value))
            
            elif op_type == 'select':
                columns = op.get('columns', [])
                result = result.select(columns)
            
            elif op_type == 'sort':
                column = op.get('column')
                ascending = op.get('ascending', True)
                result = result.sort(column, reverse=not ascending)
            
            elif op_type == 'group_by':
                columns = op.get('columns', [])
                aggs = op.get('aggregations', {})
                
                agg_exprs = []
                for col, agg in aggs.items():
                    if agg == 'sum':
                        agg_exprs.append(pl.sum(col).alias(f"{col}_sum"))
                    elif agg == 'mean':
                        agg_exprs.append(pl.mean(col).alias(f"{col}_mean"))
                    elif agg == 'min':
                        agg_exprs.append(pl.min(col).alias(f"{col}_min"))
                    elif agg == 'max':
                        agg_exprs.append(pl.max(col).alias(f"{col}_max"))
                    elif agg == 'count':
                        agg_exprs.append(pl.count().alias(f"{col}_count"))
                
                result = result.group_by(columns).agg(agg_exprs)
            
            elif op_type == 'join':
                other_df = op.get('dataframe')
                on = op.get('on')
                how = op.get('how', 'inner')
                result = result.join(other_df, on=on, how=how)
        
        return result