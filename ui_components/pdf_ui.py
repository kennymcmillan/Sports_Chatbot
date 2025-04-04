"""
PDF UI Module for Multi-Source Julius AI Chatbot.

This module provides UI components for PDF document processing.
"""

import streamlit as st
import pandas as pd
from data_sources.pdf_processor import PDFProcessor
from data_sources.dataset_handler import DatasetHandler

def render_pdf_sidebar():
    """
    Render the PDF document processing UI in the sidebar.
    
    This function displays UI elements for:
    - Uploading PDF documents
    - Extracting text and tables from PDFs
    - Viewing PDF content
    - Loading PDF content as datasets
    """
    st.markdown("### PDF Document Processing")
    
    # Initialize PDF processor if not already in session state
    if 'pdf_processor' not in st.session_state:
        st.session_state['pdf_processor'] = PDFProcessor()
    
    # Get PDF processor from session state
    pdf_processor = st.session_state['pdf_processor']
    
    # Initialize dataset handler if not already in session state
    if 'dataset_handler' not in st.session_state:
        st.session_state['dataset_handler'] = DatasetHandler()
    
    # Get dataset handler from session state
    dataset_handler = st.session_state['dataset_handler']
    
    # Option to upload a PDF document
    uploaded_pdf = st.file_uploader(
        "Upload a PDF document",
        type=["pdf"],
        key="pdf_upload"
    )
    
    # Process the uploaded PDF
    if uploaded_pdf is not None:
        with st.spinner("Processing PDF..."):
            if pdf_processor.load_pdf(uploaded_pdf):
                st.session_state['pdf_loaded'] = True
                st.success(f"PDF '{uploaded_pdf.name}' loaded successfully.")
            else:
                st.session_state['pdf_loaded'] = False
                st.error("Failed to load PDF.")
    
    # PDF processing options (only show if PDF is loaded)
    if st.session_state.get('pdf_loaded', False):
        st.markdown("### PDF Processing Options")
        
        # Display PDF metadata
        metadata = pdf_processor.get_pdf_metadata()
        if metadata:
            with st.expander("PDF Metadata", expanded=False):
                for key, value in metadata.items():
                    st.write(f"**{key.capitalize()}:** {value}")
        
        # Display PDF summary
        summary = pdf_processor.get_pdf_summary()
        if summary:
            with st.expander("PDF Summary", expanded=False):
                st.text(summary)
        
        # Option to extract text
        if st.button("Extract Text"):
            if pdf_processor.current_pdf_text:
                # Display a sample of the extracted text
                with st.expander("Extracted Text Sample", expanded=True):
                    st.text(pdf_processor.current_pdf_text[:1000] + "...")
                
                # Option to load text as dataset
                if st.button("Load Text as Dataset"):
                    # Convert PDF text to DataFrame
                    df = pdf_processor.convert_pdf_to_dataframe()
                    
                    if not df.empty:
                        # Generate a name for the dataset
                        import os
                        pdf_name = os.path.splitext(uploaded_pdf.name)[0]
                        dataset_name = f"{pdf_name}_text.csv"
                        
                        # Load the dataset
                        if dataset_handler.load_dataset_from_pdf(df, dataset_name):
                            # Get dataset from handler
                            df, name, info = dataset_handler.get_dataset_for_julius()
                            
                            # Update session state
                            st.session_state['dataset'] = df
                            st.session_state['dataset_name'] = name
                            st.session_state['dataset_source'] = 'pdf'
                            st.session_state['using_processed_dataset'] = False
                            
                            # Switch to dataset mode
                            st.session_state['current_mode'] = 'dataset'
                            st.success(f"PDF text loaded as dataset: {name}")
                            st.info("Switched to Dataset mode for analysis.")
                            
                            # Add a rerun button to refresh the UI
                            st.button("Refresh UI")
                        else:
                            st.error("Failed to load PDF text as dataset.")
            else:
                st.warning("No text extracted from PDF.")
        
        # Option to extract tables
        if st.button("Extract Tables"):
            with st.spinner("Extracting tables..."):
                tables = pdf_processor.extract_tables_from_pdf()
                
                if tables:
                    st.success(f"Extracted {len(tables)} tables from PDF.")
                    
                    # Display tables
                    for i, table in enumerate(tables):
                        with st.expander(f"Table {i+1}", expanded=i==0):
                            st.dataframe(table)
                    
                    # Option to load tables as dataset
                    if st.button("Load Tables as Dataset"):
                        # Combine all tables into one DataFrame
                        import pandas as pd
                        combined_df = pd.concat(tables, ignore_index=True)
                        
                        # Generate a name for the dataset
                        import os
                        pdf_name = os.path.splitext(uploaded_pdf.name)[0]
                        dataset_name = f"{pdf_name}_tables.csv"
                        
                        # Load the dataset
                        if dataset_handler.load_dataset_from_pdf(combined_df, dataset_name):
                            # Get dataset from handler
                            df, name, info = dataset_handler.get_dataset_for_julius()
                            
                            # Update session state
                            st.session_state['dataset'] = df
                            st.session_state['dataset_name'] = name
                            st.session_state['dataset_source'] = 'pdf'
                            st.session_state['using_processed_dataset'] = False
                            
                            # Switch to dataset mode
                            st.session_state['current_mode'] = 'dataset'
                            st.success(f"PDF tables loaded as dataset: {name}")
                            st.info("Switched to Dataset mode for analysis.")
                            
                            # Add a rerun button to refresh the UI
                            st.button("Refresh UI")
                        else:
                            st.error("Failed to load PDF tables as dataset.")
                else:
                    st.warning("No tables found in PDF.")
        
        # Button to clear PDF
        if st.button("Clear PDF"):
            pdf_processor.cleanup()
            st.session_state['pdf_loaded'] = False
            st.success("PDF cleared.")