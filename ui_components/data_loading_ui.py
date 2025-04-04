"""
Data Loading UI Module for Multi-Source Julius AI Chatbot.

This module provides UI components for loading data from various sources:
- File uploads (CSV, Excel, etc.)
- PDF documents
"""

import os
import streamlit as st
import pandas as pd
from data_sources.dataset_handler import DatasetHandler
from data_sources.pdf_processor import PDFProcessor

def render_data_loading_sidebar():
    """
    Render the data loading UI in the sidebar.
    
    This function displays UI elements for:
    - Loading datasets from file uploads
    - Loading data from PDF documents
    """
    st.markdown("### Data Loading")
    
    # Initialize dataset handler if not already in session state
    if 'dataset_handler' not in st.session_state:
        st.session_state['dataset_handler'] = DatasetHandler()
    
    # Get dataset handler from session state
    dataset_handler = st.session_state['dataset_handler']
    
    # Initialize PDF processor if not already in session state
    if 'pdf_processor' not in st.session_state:
        st.session_state['pdf_processor'] = PDFProcessor()
    
    # Get PDF processor from session state
    pdf_processor = st.session_state['pdf_processor']
    
    # Create tabs for different data loading methods
    data_source_tab1, data_source_tab2 = st.tabs(["File Upload", "PDF Document"])
    
    # File Upload Tab
    with data_source_tab1:
        st.markdown("#### Upload Dataset")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload a dataset file (CSV, Excel, etc.)",
            type=["csv", "xlsx", "xls", "json", "parquet", "feather", "pickle", "pkl"]
        )
        
        # Process uploaded file
        if uploaded_file is not None:
            if st.button("Load Dataset"):
                with st.spinner("Loading dataset..."):
                    # Load dataset from uploaded file
                    if dataset_handler.load_dataset_from_file(uploaded_file):
                        # Get dataset from handler
                        df, name, info = dataset_handler.get_dataset_for_julius()
                        
                        # Update session state
                        st.session_state['dataset'] = df
                        st.session_state['dataset_name'] = name
                        st.session_state['dataset_source'] = 'upload'
                        st.session_state['dataset_loaded'] = True
                        
                        # Display success message
                        st.success(f"Dataset loaded: {name}")
                        
                        # Display dataset info
                        st.markdown("#### Dataset Info")
                        st.write(f"Rows: {info.get('shape', (0, 0))[0]}")
                        st.write(f"Columns: {info.get('shape', (0, 0))[1]}")
                        
                        # Display column names
                        if 'columns' in info:
                            with st.expander("Column Names", expanded=False):
                                st.write(", ".join(info['columns']))
                    else:
                        st.error("Failed to load dataset.")
    
    # PDF Document Tab
    with data_source_tab2:
        st.markdown("#### Upload PDF Document")
        
        # PDF file uploader
        pdf_file = st.file_uploader(
            "Upload a PDF document",
            type=["pdf"]
        )
        
        # Process uploaded PDF
        if pdf_file is not None:
            # PDF processing options
            st.markdown("#### PDF Processing Options")
            
            # Extract tables option
            extract_tables = st.checkbox("Extract tables from PDF", value=True)
            
            # Extract text option
            extract_text = st.checkbox("Extract text from PDF", value=True)
            
            # Process PDF button
            if st.button("Process PDF"):
                with st.spinner("Processing PDF..."):
                    # Process PDF
                    success, tables, text = pdf_processor.process_pdf(pdf_file, extract_tables, extract_text)
                    
                    if success:
                        # Store PDF processing results in session state
                        st.session_state['pdf_tables'] = tables
                        st.session_state['pdf_text'] = text
                        st.session_state['pdf_loaded'] = True
                        st.session_state['pdf_name'] = pdf_file.name
                        
                        # Display success message
                        st.success(f"PDF processed: {pdf_file.name}")
                        
                        # Display extracted tables
                        if extract_tables and tables:
                            st.markdown("#### Extracted Tables")
                            st.write(f"Found {len(tables)} tables")
                            
                            # Display tables in expander
                            with st.expander("View Tables", expanded=False):
                                for i, table in enumerate(tables):
                                    st.markdown(f"**Table {i+1}**")
                                    st.dataframe(table)
                            
                            # Option to select a table as dataset
                            if len(tables) > 0:
                                selected_table = st.selectbox(
                                    "Select a table to use as dataset",
                                    range(len(tables)),
                                    format_func=lambda x: f"Table {x+1}"
                                )
                                
                                if st.button("Use Selected Table as Dataset"):
                                    # Load selected table as dataset
                                    if dataset_handler.load_dataset_from_pdf(tables[selected_table], f"{pdf_file.name}_table_{selected_table+1}.csv"):
                                        # Get dataset from handler
                                        df, name, info = dataset_handler.get_dataset_for_julius()
                                        
                                        # Update session state
                                        st.session_state['dataset'] = df
                                        st.session_state['dataset_name'] = name
                                        st.session_state['dataset_source'] = 'pdf'
                                        st.session_state['dataset_loaded'] = True
                                        
                                        # Display success message
                                        st.success(f"Table loaded as dataset: {name}")
                                    else:
                                        st.error("Failed to load table as dataset.")
                        
                        # Display extracted text
                        if extract_text and text:
                            with st.expander("View Extracted Text", expanded=False):
                                st.markdown(text)
                    else:
                        st.error("Failed to process PDF.")
    
    # Display current dataset info if loaded
    if st.session_state.get('dataset_loaded', False) and st.session_state.get('dataset') is not None:
        st.markdown("---")
        st.markdown("### Current Dataset")
        st.write(f"Name: {st.session_state.get('dataset_name', 'Unknown')}")
        st.write(f"Source: {st.session_state.get('dataset_source', 'Unknown')}")
        st.write(f"Shape: {st.session_state.get('dataset').shape}")
        
        # Option to view dataset
        with st.expander("View Dataset", expanded=False):
            st.dataframe(st.session_state.get('dataset').head(10))
        
        # Option to clear dataset
        if st.button("Clear Dataset"):
            st.session_state['dataset'] = None
            st.session_state['dataset_name'] = None
            st.session_state['dataset_source'] = None
            st.session_state['dataset_loaded'] = False
            st.success("Dataset cleared.")
            st.experimental_rerun()