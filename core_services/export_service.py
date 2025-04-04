"""
Export Service Module for Multi-Source Julius AI Chatbot.

This module provides the ExportService class for handling data export in various formats.
It provides methods for exporting data to CSV, Excel, JSON, Parquet, and other formats.
"""

import os
import io
import zipfile
import tempfile
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path

import polars as pl
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field

class ExportOptions(BaseModel):
    """Export options model."""
    include_index: bool = Field(False, description="Whether to include index in export")
    date_format: Optional[str] = Field(None, description="Date format for date columns")
    float_format: Optional[str] = Field(None, description="Float format for numeric columns")
    encoding: str = Field("utf-8", description="Encoding for text-based formats")
    compression: Optional[str] = Field(None, description="Compression method")
    sheet_name: str = Field("Sheet1", description="Sheet name for Excel export")

class ExportService:
    """
    Service for handling data export in various formats.
    
    This service provides methods for:
    - Exporting data to CSV, Excel, JSON, Parquet, and other formats
    - Compressing exported data
    - Generating download links
    """
    
    def __init__(self, export_dir: str = "exports"):
        """
        Initialize the ExportService.
        
        Args:
            export_dir: Directory for storing exported files
        """
        self.export_dir = export_dir
        os.makedirs(export_dir, exist_ok=True)
    
    def export_csv(self, df: Union[pl.DataFrame, pd.DataFrame], options: Optional[ExportOptions] = None) -> bytes:
        """
        Export data to CSV format.
        
        Args:
            df: Polars or Pandas DataFrame
            options: Export options
            
        Returns:
            bytes: CSV data as bytes
        """
        options = options or ExportOptions()
        
        # Convert Polars DataFrame to Pandas if needed
        if isinstance(df, pl.DataFrame):
            pandas_df = df.to_pandas()
        else:
            pandas_df = df
        
        # Export to CSV
        buffer = io.StringIO()
        pandas_df.to_csv(
            buffer,
            index=options.include_index,
            date_format=options.date_format,
            float_format=options.float_format,
            encoding=options.encoding
        )
        
        return buffer.getvalue().encode(options.encoding)
    
    def export_excel(self, df: Union[pl.DataFrame, pd.DataFrame], options: Optional[ExportOptions] = None) -> bytes:
        """
        Export data to Excel format.
        
        Args:
            df: Polars or Pandas DataFrame
            options: Export options
            
        Returns:
            bytes: Excel data as bytes
        """
        options = options or ExportOptions()
        
        # Convert Polars DataFrame to Pandas if needed
        if isinstance(df, pl.DataFrame):
            pandas_df = df.to_pandas()
        else:
            pandas_df = df
        
        # Export to Excel
        buffer = io.BytesIO()
        pandas_df.to_excel(
            buffer,
            sheet_name=options.sheet_name,
            index=options.include_index,
            float_format=options.float_format,
            engine='openpyxl'
        )
        
        buffer.seek(0)
        return buffer.getvalue()
    
    def export_json(self, df: Union[pl.DataFrame, pd.DataFrame], options: Optional[ExportOptions] = None) -> bytes:
        """
        Export data to JSON format.
        
        Args:
            df: Polars or Pandas DataFrame
            options: Export options
            
        Returns:
            bytes: JSON data as bytes
        """
        options = options or ExportOptions()
        
        # Convert Polars DataFrame to Pandas if needed
        if isinstance(df, pl.DataFrame):
            pandas_df = df.to_pandas()
        else:
            pandas_df = df
        
        # Export to JSON
        json_str = pandas_df.to_json(
            orient='records',
            date_format='iso',
            indent=2
        )
        
        return json_str.encode(options.encoding)
    
    def export_parquet(self, df: Union[pl.DataFrame, pd.DataFrame], options: Optional[ExportOptions] = None) -> bytes:
        """
        Export data to Parquet format.
        
        Args:
            df: Polars or Pandas DataFrame
            options: Export options
            
        Returns:
            bytes: Parquet data as bytes
        """
        options = options or ExportOptions()
        
        # Use Polars for Parquet export
        if isinstance(df, pd.DataFrame):
            polars_df = pl.from_pandas(df)
        else:
            polars_df = df
        
        # Export to Parquet
        buffer = io.BytesIO()
        polars_df.write_parquet(buffer, compression=options.compression or "snappy")
        
        buffer.seek(0)
        return buffer.getvalue()
    
    def export_multiple_formats(self, df: Union[pl.DataFrame, pd.DataFrame], 
                               formats: List[str], options: Optional[ExportOptions] = None) -> bytes:
        """
        Export data to multiple formats and create a ZIP archive.
        
        Args:
            df: Polars or Pandas DataFrame
            formats: List of formats to export ('csv', 'excel', 'json', 'parquet')
            options: Export options
            
        Returns:
            bytes: ZIP archive as bytes
        """
        options = options or ExportOptions()
        
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Export to each format
            for fmt in formats:
                if fmt.lower() == 'csv':
                    data = self.export_csv(df, options)
                    file_path = os.path.join(temp_dir, "data.csv")
                elif fmt.lower() == 'excel':
                    data = self.export_excel(df, options)
                    file_path = os.path.join(temp_dir, "data.xlsx")
                elif fmt.lower() == 'json':
                    data = self.export_json(df, options)
                    file_path = os.path.join(temp_dir, "data.json")
                elif fmt.lower() == 'parquet':
                    data = self.export_parquet(df, options)
                    file_path = os.path.join(temp_dir, "data.parquet")
                else:
                    continue
                
                # Write to file
                with open(file_path, 'wb') as f:
                    f.write(data)
            
            # Create ZIP archive
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for file_name in os.listdir(temp_dir):
                    file_path = os.path.join(temp_dir, file_name)
                    zip_file.write(file_path, file_name)
            
            buffer.seek(0)
            return buffer.getvalue()
    
    def save_export(self, data: bytes, file_name: str) -> str:
        """
        Save exported data to a file.
        
        Args:
            data: Exported data as bytes
            file_name: File name
            
        Returns:
            str: Path to the saved file
        """
        # Create full path
        file_path = os.path.join(self.export_dir, file_name)
        
        # Save data
        with open(file_path, 'wb') as f:
            f.write(data)
        
        return file_path
    
    def export_visualization(self, fig, file_name: str, format: str = 'png', dpi: int = 300) -> str:
        """
        Export a matplotlib figure to an image file.
        
        Args:
            fig: Matplotlib figure
            file_name: File name
            format: Image format ('png', 'jpg', 'svg', 'pdf')
            dpi: Resolution in dots per inch
            
        Returns:
            str: Path to the saved file
        """
        # Create full path
        file_path = os.path.join(self.export_dir, file_name)
        
        # Save figure
        fig.savefig(file_path, format=format, dpi=dpi, bbox_inches='tight')
        
        return file_path
    
    def export_report(self, title: str, content: str, visualizations: List[str] = None, 
                     format: str = 'html') -> bytes:
        """
        Export a report with text and visualizations.
        
        Args:
            title: Report title
            content: Report content (Markdown or HTML)
            visualizations: List of paths to visualization images
            format: Report format ('html', 'pdf', 'md')
            
        Returns:
            bytes: Report data as bytes
        """
        if format.lower() == 'html':
            # Create HTML report
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>{title}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                    h1 {{ color: #333; }}
                    img {{ max-width: 100%; height: auto; margin: 20px 0; }}
                </style>
            </head>
            <body>
                <h1>{title}</h1>
                {content}
            """
            
            # Add visualizations
            if visualizations:
                for i, viz_path in enumerate(visualizations):
                    if os.path.exists(viz_path):
                        # Get file name
                        viz_name = os.path.basename(viz_path)
                        html += f"""
                        <div class="visualization">
                            <h3>Visualization {i+1}</h3>
                            <img src="{viz_name}" alt="Visualization {i+1}">
                        </div>
                        """
            
            html += """
            </body>
            </html>
            """
            
            return html.encode('utf-8')
        
        elif format.lower() == 'md':
            # Create Markdown report
            md = f"# {title}\n\n{content}\n\n"
            
            # Add visualizations
            if visualizations:
                md += "## Visualizations\n\n"
                for i, viz_path in enumerate(visualizations):
                    if os.path.exists(viz_path):
                        # Get file name
                        viz_name = os.path.basename(viz_path)
                        md += f"### Visualization {i+1}\n\n![Visualization {i+1}]({viz_name})\n\n"
            
            return md.encode('utf-8')
        
        else:
            raise ValueError(f"Unsupported report format: {format}")