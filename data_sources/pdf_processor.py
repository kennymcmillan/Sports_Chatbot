"""
PDF Processor Module for Julius AI Chatbot.

This module provides functionality to load, process, and extract text from PDF documents.
"""

import os
import tempfile
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import streamlit as st

# Try to import PDF processing libraries
try:
    import pypdf
    from langchain.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    LANGCHAIN_PDF_AVAILABLE = True
except ImportError:
    LANGCHAIN_PDF_AVAILABLE = False


class PDFProcessor:
    """
    A class to handle PDF document loading and processing.
    
    This class provides methods to:
    - Load PDF documents
    - Extract text from PDFs
    - Process and chunk PDF content
    - Convert PDF content to structured data
    """
    
    def __init__(self):
        """Initialize the PDF Processor."""
        self.current_pdf = None
        self.current_pdf_path = None
        self.current_pdf_text = None
        self.current_pdf_chunks = None
        self.temp_dir = tempfile.mkdtemp()
        
        # Check if PDF processing libraries are available
        if not LANGCHAIN_PDF_AVAILABLE:
            st.warning("PDF processing libraries are not installed. PDF functionality will be limited.")
    
    def load_pdf(self, pdf_file) -> bool:
        """
        Load a PDF file.
        
        Args:
            pdf_file: Uploaded PDF file from Streamlit
            
        Returns:
            bool: True if loading successful, False otherwise
        """
        if not LANGCHAIN_PDF_AVAILABLE:
            st.error("PDF processing libraries are not installed. Cannot process PDF.")
            return False
            
        try:
            # Save the uploaded file to a temporary location
            temp_pdf_path = os.path.join(self.temp_dir, pdf_file.name)
            with open(temp_pdf_path, "wb") as f:
                f.write(pdf_file.getbuffer())
            
            # Load the PDF using PyPDFLoader
            loader = PyPDFLoader(temp_pdf_path)
            self.current_pdf = loader.load()
            self.current_pdf_path = temp_pdf_path
            
            # Extract text from the PDF
            self.current_pdf_text = self._extract_text_from_pdf()
            
            # Chunk the PDF content
            self.current_pdf_chunks = self._chunk_pdf_content()
            
            return True
            
        except Exception as e:
            st.error(f"Error loading PDF: {str(e)}")
            return False
    
    def _extract_text_from_pdf(self) -> str:
        """
        Extract text from the loaded PDF.
        
        Returns:
            str: Extracted text
        """
        if not self.current_pdf:
            return ""
            
        # Combine text from all pages
        return "\n\n".join([page.page_content for page in self.current_pdf])
    
    def _chunk_pdf_content(self) -> List[str]:
        """
        Chunk the PDF content for processing.
        
        Returns:
            List[str]: Chunked content
        """
        if not self.current_pdf_text:
            return []
            
        # Create a text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        # Split the text into chunks
        return text_splitter.split_text(self.current_pdf_text)
    
    def get_pdf_metadata(self) -> Dict[str, Any]:
        """
        Get metadata from the loaded PDF.
        
        Returns:
            Dict[str, Any]: PDF metadata
        """
        if not self.current_pdf_path:
            return {}
            
        try:
            # Open the PDF file
            with open(self.current_pdf_path, "rb") as f:
                pdf = pypdf.PdfReader(f)
                
                # Extract metadata
                metadata = {
                    "title": pdf.metadata.get("/Title", ""),
                    "author": pdf.metadata.get("/Author", ""),
                    "subject": pdf.metadata.get("/Subject", ""),
                    "creator": pdf.metadata.get("/Creator", ""),
                    "producer": pdf.metadata.get("/Producer", ""),
                    "creation_date": pdf.metadata.get("/CreationDate", ""),
                    "modification_date": pdf.metadata.get("/ModDate", ""),
                    "num_pages": len(pdf.pages)
                }
                
                return metadata
                
        except Exception as e:
            st.error(f"Error getting PDF metadata: {str(e)}")
            return {}
    
    def extract_tables_from_pdf(self) -> List[pd.DataFrame]:
        """
        Extract tables from the loaded PDF.
        
        Returns:
            List[pd.DataFrame]: Extracted tables
        """
        if not self.current_pdf_path:
            return []
            
        try:
            # Try to import tabula-py for table extraction
            try:
                import tabula
                TABULA_AVAILABLE = True
            except ImportError:
                TABULA_AVAILABLE = False
                st.warning("tabula-py is not installed. Cannot extract tables from PDF.")
                return []
            
            if not TABULA_AVAILABLE:
                return []
                
            # Extract tables using tabula
            tables = tabula.read_pdf(self.current_pdf_path, pages='all', multiple_tables=True)
            
            # Clean up tables
            cleaned_tables = []
            for table in tables:
                # Remove empty rows and columns
                table = table.dropna(how='all').dropna(axis=1, how='all')
                if not table.empty:
                    cleaned_tables.append(table)
            
            return cleaned_tables
            
        except Exception as e:
            st.error(f"Error extracting tables from PDF: {str(e)}")
            return []
    
    def convert_pdf_to_dataframe(self) -> pd.DataFrame:
        """
        Convert PDF content to a DataFrame.
        
        Returns:
            pd.DataFrame: PDF content as a DataFrame
        """
        if not self.current_pdf_chunks:
            return pd.DataFrame()
            
        # Create a DataFrame with chunks and page numbers
        df = pd.DataFrame({
            'chunk_id': range(len(self.current_pdf_chunks)),
            'content': self.current_pdf_chunks
        })
        
        return df
    
    def get_pdf_summary(self) -> str:
        """
        Generate a summary of the PDF content.
        
        Returns:
            str: PDF summary
        """
        if not self.current_pdf_text:
            return "No PDF loaded"
            
        # Get basic statistics
        num_chars = len(self.current_pdf_text)
        num_words = len(self.current_pdf_text.split())
        num_chunks = len(self.current_pdf_chunks) if self.current_pdf_chunks else 0
        
        # Get metadata
        metadata = self.get_pdf_metadata()
        
        # Create summary
        summary = f"PDF Summary:\n"
        summary += f"- File: {os.path.basename(self.current_pdf_path)}\n"
        summary += f"- Pages: {metadata.get('num_pages', 'Unknown')}\n"
        summary += f"- Characters: {num_chars}\n"
        summary += f"- Words: {num_words}\n"
        summary += f"- Chunks: {num_chunks}\n"
        
        # Add metadata if available
        if metadata.get('title'):
            summary += f"- Title: {metadata['title']}\n"
        if metadata.get('author'):
            summary += f"- Author: {metadata['author']}\n"
        if metadata.get('subject'):
            summary += f"- Subject: {metadata['subject']}\n"
        
        return summary
    
    def cleanup(self) -> None:
        """
        Clean up temporary files.
        """
        try:
            # Remove temporary directory and its contents
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception:
            pass