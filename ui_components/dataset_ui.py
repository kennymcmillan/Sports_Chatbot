"""
Dataset UI Module for Multi-Source Julius AI Chatbot.

This module provides UI components for dataset upload and management.
"""

import os
import streamlit as st
import pandas as pd
from data_sources.dataset_handler import DatasetHandler

def render_dataset_sidebar():
    """
    Render the dataset management UI in the sidebar.
    
    This function displays UI elements for:
    - Uploading datasets
    - Displaying dataset information
    - Cleaning datasets
    - Clearing datasets
    """
    st.markdown("### Dataset Management")
    
    # Initialize dataset handler if not already in session state
    if 'dataset_handler' not in st.session_state:
        st.session_state['dataset_handler'] = DatasetHandler()
    
    # Get dataset handler from session state
    dataset_handler = st.session_state['dataset_handler']
    
    # Option to upload a dataset
    uploaded_dataset = st.file_uploader(
        "Upload a dataset",
        type=["csv", "xlsx", "json", "parquet", "feather", "pickle"],
        key="dataset_upload"
    )
    
    # Process the uploaded dataset
    if uploaded_dataset is not None and (st.session_state['dataset'] is None or
                                        uploaded_dataset.name != st.session_state.get('dataset_name')):
        if dataset_handler.load_dataset_from_file(uploaded_dataset):
            # Get dataset from handler
            df, name, info = dataset_handler.get_dataset_for_julius()
            
            # Update session state
            st.session_state['dataset'] = df
            st.session_state['dataset_name'] = name
            st.session_state['dataset_source'] = 'upload'
            st.session_state['using_processed_dataset'] = False
            st.success(f"Dataset '{name}' loaded successfully.")
        else:
            st.error("Failed to load dataset.")
    
    # Display current dataset info
    if st.session_state['dataset'] is not None:
        st.write(f"Current Dataset: {st.session_state['dataset_name']}")
        st.write(f"Source: {st.session_state['dataset_source']}")
        
        # Display dataset shape
        st.write(f"Shape: {st.session_state['dataset'].shape[0]} rows × {st.session_state['dataset'].shape[1]} columns")
        
        # Option to view dataset sample
        with st.expander("View Dataset Sample", expanded=False):
            st.dataframe(st.session_state['dataset'].head(5))
        
        # Option to clean dataset
        if st.button("Clean Dataset"):
            if dataset_handler.clean_dataset():
                # Get cleaned dataset
                df, name, info = dataset_handler.get_dataset_for_julius()
                
                # Update session state
                st.session_state['dataset'] = df
                st.session_state['dataset_name'] = name
                st.session_state['dataset_source'] = 'cleaned'
                st.success("Dataset cleaned successfully.")
            else:
                st.error("Failed to clean dataset.")
    
    # Button to clear dataset
    if st.session_state['dataset'] is not None:
        if st.button("Clear Dataset"):
            dataset_handler.clear_dataset()
            st.session_state['dataset'] = None
            st.session_state['dataset_name'] = None
            st.session_state['dataset_source'] = None
            st.session_state['dataset_loaded'] = False
            st.session_state['cached_dataset'] = None
            st.session_state['using_processed_dataset'] = False
            st.success("Dataset cleared.")
    
    # Check for processed dataset
    processed_dataset_path = "temp_files/current_dataset.csv"
    if os.path.exists(processed_dataset_path) and not st.session_state.get('using_processed_dataset', False):
        if st.button("Use Processed Dataset"):
            try:
                # Load the processed dataset
                processed_df = pd.read_csv(processed_dataset_path)
                
                # Update dataset handler
                dataset_handler.load_dataset_from_sql(processed_df, "current_dataset.csv")
                
                # Get dataset from handler
                df, name, info = dataset_handler.get_dataset_for_julius()
                
                # Update session state
                st.session_state['dataset'] = df
                st.session_state['dataset_name'] = name
                st.session_state['dataset_source'] = 'processed'
                st.session_state['using_processed_dataset'] = True
                st.session_state['dataset_loaded'] = True
                st.session_state['cached_dataset'] = df
                st.success("Loaded processed dataset with enhanced features.")
            except Exception as e:
                st.error(f"Error loading processed dataset: {e}")