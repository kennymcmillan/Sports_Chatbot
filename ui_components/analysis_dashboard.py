"""
Analysis Dashboard UI Module for Multi-Source Julius AI Chatbot.

This module provides UI components for analyzing datasets using Julius AI.
"""

import os
import streamlit as st
import pandas as pd
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, Dict, List, Any, Tuple

from core_services.database_service import DatabaseService
import time
import glob
import re
import json
from pathlib import Path

from core_services.data_service import DataService
from core_services.ai_service import AIService, DataAnalysisRequest, DataAnalysisResponse
from core_services.export_service import ExportService

def render_analysis_dashboard_sidebar(data_service: DataService, database_service: Optional[DatabaseService] = None):
    """
    Render the analysis dashboard sidebar.
    
    Args:
        data_service: DataService instance
        database_service: DatabaseService instance (optional)
    """
    # Add selector for sidebar
    sidebar_option = st.radio("Analysis Type", ["Dataset Analysis", "Database Reasoning"])
    
    if sidebar_option == "Dataset Analysis":
        render_dataset_analysis_sidebar(data_service)
    else:  # Database Reasoning
        if database_service:
            # Import database reasoning UI
            from ui_components.database_reasoning_ui import render_database_reasoning_sidebar
            
            # Render database reasoning sidebar
            render_database_reasoning_sidebar(database_service)
        else:
            st.warning("Database service not available.")

def render_dataset_analysis_sidebar(data_service: DataService):
    """
    Render the dataset analysis sidebar.
    
    Args:
        data_service: DataService instance
    """
    st.markdown("### Analysis Dashboard")
    
    # Check if dataset is loaded
    if data_service.current_dataset is None:
        st.warning("No dataset loaded. Please load a dataset first.")
        return
    
    # Display current dataset info
    st.markdown("#### Current Dataset")
    st.write(f"Name: {data_service.current_dataset_name}")
    st.write(f"Source: {data_service.current_dataset_source}")
    st.write(f"Shape: {data_service.current_dataset.shape}")
    
    # Analysis options
    st.markdown("#### Analysis Options")
    
    # Option to generate visualizations
    st.checkbox("Generate visualizations", value=True, key="generate_visualizations")
    
    # Option to generate code
    st.checkbox("Generate code", value=True, key="generate_code")
    
    # Option to execute generated code
    st.checkbox("Execute generated code", value=True, key="execute_code")
    
    # Analysis history
    if 'analysis_history' in st.session_state and st.session_state['analysis_history']:
        st.markdown("#### Analysis History")
        
        for i, (query, _) in enumerate(st.session_state['analysis_history']):
            if st.button(f"{query[:30]}...", key=f"history_{i}"):
                st.session_state['selected_analysis'] = i
                st.experimental_rerun()

def render_analysis_dashboard(data_service: DataService, ai_service: AIService, database_service: DatabaseService, export_service: ExportService):
    """
    Render the analysis dashboard main content.
    
    Args:
        data_service: DataService instance
        ai_service: AIService instance
        database_service: DatabaseService instance
        export_service: ExportService instance
    """
    st.header("Analysis Dashboard")
    
    # Create tabs for different analysis methods
    tab1, tab2 = st.tabs(["Dataset Analysis", "Database Reasoning"])
    
    with tab1:
        render_dataset_analysis_tab(data_service, ai_service, export_service)
    
    with tab2:
        # Import database reasoning UI
        from ui_components.database_reasoning_ui import render_database_reasoning_sidebar, render_database_reasoning_ui
        
        # Initialize SQL connector if not already in session state
        if 'sql_connector' not in st.session_state:
            # Get OpenAI API key from environment
            import os
            openai_api_key = os.getenv("OPENAI_API_KEY")
            from data_sources.sql_connector import SQLConnector
            st.session_state['sql_connector'] = SQLConnector(api_key=openai_api_key)
        
        # Get SQL connector from session state
        sql_connector = st.session_state['sql_connector']
        
        # Initialize database reasoning if not already in session state
        if 'db_reasoning' not in st.session_state:
            from data_sources.db_reasoning import DatabaseReasoning
            st.session_state['db_reasoning'] = DatabaseReasoning(
                sql_connector=sql_connector,
                julius=ai_service.julius if hasattr(ai_service, 'julius') else st.session_state.get('julius')
            )
        
        # Render database reasoning UI
        render_database_reasoning_ui(database_service, ai_service, data_service, export_service)

def render_dataset_analysis_tab(data_service: DataService, ai_service: AIService, export_service: ExportService):
    """
    Render the dataset analysis tab.
    
    Args:
        data_service: DataService instance
        ai_service: AIService instance
        export_service: ExportService instance
    """
    # Check if dataset is loaded
    if data_service.current_dataset is None:
        st.info("No dataset loaded. Please load a dataset first.")
        return
    
    # Get current dataset
    df = data_service.current_dataset
    
    # Initialize analysis history in session state
    if 'analysis_history' not in st.session_state:
        st.session_state['analysis_history'] = []
    
    # Initialize selected analysis in session state
    if 'selected_analysis' not in st.session_state:
        st.session_state['selected_analysis'] = None
    
    # Analysis query input
    st.markdown("### Ask a question about your data")
    analysis_query = st.text_area(
        "Enter your analysis query",
        placeholder="Example: What is the correlation between age and income? Show me a visualization.",
        height=100,
        key="analysis_query"
    )
    
    # Analyze button
    if st.button("Analyze", key="analyze_button"):
        if analysis_query:
            with st.spinner("Analyzing data..."):
                try:
                    # Get dataset info
                    dataset_info = f"""
                    Dataset Name: {data_service.current_dataset_name}
                    Dataset Source: {data_service.current_dataset_source}
                    Shape: {df.height} rows x {df.width} columns
                    
                    Columns: {', '.join(df.columns)}
                    
                    Sample Data:
                    {df.head(5).to_pandas().to_string()}
                    
                    Data Types:
                    {pd.DataFrame({'Column': df.columns, 'Type': [str(df[col].dtype) for col in df.columns]}).to_string(index=False)}
                    """
                    
                    # Get analysis options
                    generate_visualizations = st.session_state.get('generate_visualizations', True)
                    generate_code = st.session_state.get('generate_code', True)
                    
                    # Create analysis request
                    request = DataAnalysisRequest(
                        query=analysis_query,
                        data_info=dataset_info,
                        generate_code=generate_code,
                        generate_visualizations=generate_visualizations
                    )
                    
                    # Analyze data
                    response = ai_service.analyze_data(request)
                    
                    # Add to analysis history
                    st.session_state['analysis_history'].append((analysis_query, response))
                    
                    # Set as selected analysis
                    st.session_state['selected_analysis'] = len(st.session_state['analysis_history']) - 1
                    
                    # Display analysis
                    display_analysis(response, data_service, export_service)
                
                except Exception as e:
                    st.error(f"Error analyzing data: {str(e)}")
        else:
            st.warning("Please enter an analysis query.")
    
    # Display selected analysis if available
    if st.session_state.get('selected_analysis') is not None:
        selected_index = st.session_state['selected_analysis']
        
        if 0 <= selected_index < len(st.session_state['analysis_history']):
            _, response = st.session_state['analysis_history'][selected_index]
            display_analysis(response, data_service, export_service)

def display_analysis(response: DataAnalysisResponse, data_service: DataService, export_service: ExportService):
    """
    Display analysis results.
    
    Args:
        response: Analysis response
        data_service: DataService instance
        export_service: ExportService instance
    """
    # Display analysis text
    st.markdown("### Analysis Results")
    st.markdown(response.analysis)
    
    # Display images if available
    if response.image_urls:
        st.markdown("### Visualizations")
        
        for i, url in enumerate(response.image_urls):
            try:
                st.image(url, caption=f"Visualization {i+1}")
            except Exception as e:
                st.error(f"Error displaying image: {str(e)}")
    
    # Display web context if available
    if response.web_context:
        st.markdown("### Additional Context")
        with st.expander("View Additional Context", expanded=True):
            st.markdown(response.web_context)
    
    # Display code if available
    if response.code:
        st.markdown("### Generated Code")
        
        with st.expander("View Code", expanded=False):
            st.code(response.code, language="python")
            
            # Save code to file
            code_filename = f"outputs/analysis_code_{int(time.time())}.py"
            os.makedirs("outputs", exist_ok=True)
            
            with open(code_filename, "w") as f:
                f.write(response.code)
            
            # Download button
            st.download_button(
                label="Download Code",
                data=response.code,
                file_name=os.path.basename(code_filename),
                mime="text/plain",
                key="download_code"
            )
            
            # Execute code button
            if st.session_state.get('execute_code', True) and st.button("Execute Code", key="execute_code_button"):
                with st.spinner("Executing code..."):
                    try:
                        # Save current dataset to CSV for code to use
                        dataset_path = "temp_files/current_dataset.csv"
                        os.makedirs("temp_files", exist_ok=True)
                        data_service.current_dataset.write_csv(dataset_path)
                        
                        # Create a modified version of the code that uses the current dataset
                        modified_code = response.code
                        
                        # Replace data loading code with current dataset
                        data_loading_patterns = [
                            r"pd\.read_csv\(['\"].*['\"]\)",
                            r"pd\.read_excel\(['\"].*['\"]\)",
                            r"pd\.read_parquet\(['\"].*['\"]\)",
                            r"pd\.read_json\(['\"].*['\"]\)"
                        ]
                        
                        for pattern in data_loading_patterns:
                            modified_code = re.sub(
                                pattern,
                                f'pd.read_csv("{dataset_path}")',
                                modified_code
                            )
                        
                        # Save modified code
                        modified_code_path = f"outputs/modified_code_{int(time.time())}.py"
                        with open(modified_code_path, "w") as f:
                            f.write(modified_code)
                        
                        # Execute code
                        import subprocess
                        import sys
                        
                        # Create output directory for plots
                        os.makedirs("outputs/plots", exist_ok=True)
                        
                        # Set working directory to outputs/plots
                        cwd = os.getcwd()
                        os.chdir("outputs/plots")
                        
                        # Execute code
                        result = subprocess.run(
                            [sys.executable, f"../../{modified_code_path}"],
                            capture_output=True,
                            text=True
                        )
                        
                        # Restore working directory
                        os.chdir(cwd)
                        
                        # Display output
                        if result.stdout:
                            st.subheader("Code Output:")
                            st.text(result.stdout)
                        
                        # Display errors
                        if result.stderr:
                            st.error("Code Execution Error:")
                            st.code(result.stderr)
                        
                        # Display generated plots
                        plot_files = glob.glob("outputs/plots/*.png") + glob.glob("outputs/plots/*.jpg")
                        
                        if plot_files:
                            st.subheader("Generated Plots:")
                            
                            for plot_file in plot_files:
                                st.image(plot_file, caption=os.path.basename(plot_file))
                                
                                # Export plot
                                export_service.save_export(
                                    open(plot_file, "rb").read(),
                                    os.path.basename(plot_file)
                                )
                    
                    except Exception as e:
                        st.error(f"Error executing code: {str(e)}")
    
    # Export options
    st.markdown("### Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Export analysis as Markdown
        if st.button("Export as Markdown", key="export_markdown"):
            with st.spinner("Exporting..."):
                # Create Markdown content
                md_content = f"# Analysis Results\n\n{response.analysis}\n\n"
                
                # Add code if available
                if response.code:
                    md_content += f"## Generated Code\n\n```python\n{response.code}\n```\n\n"
                
                # Add visualizations if available
                if response.visualizations:
                    md_content += "## Visualizations\n\n"
                    
                    for i, viz in enumerate(response.visualizations):
                        md_content += f"### Visualization {i+1}\n\n{viz}\n\n"
                
                # Add web context if available
                if response.web_context:
                    md_content += f"## Additional Context\n\n{response.web_context}\n\n"
                
                # Export
                md_data = md_content.encode('utf-8')
                
                st.download_button(
                    label="Download Markdown",
                    data=md_data,
                    file_name="analysis_results.md",
                    mime="text/markdown",
                    key="download_markdown"
                )
    
    with col2:
        # Export analysis as HTML
        if st.button("Export as HTML", key="export_html"):
            with st.spinner("Exporting..."):
                # Create HTML content
                html_content = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Analysis Results</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                        h1, h2, h3 {{ color: #333; }}
                        pre {{ background-color: #f5f5f5; padding: 10px; border-radius: 5px; overflow-x: auto; }}
                        img {{ max-width: 100%; height: auto; margin: 20px 0; }}
                    </style>
                </head>
                <body>
                    <h1>Analysis Results</h1>
                    <div>{response.analysis}</div>
                """
                
                # Add web context if available
                if response.web_context:
                    html_content += f"""
                    <h2>Additional Context</h2>
                    <div>{response.web_context}</div>
                    """
                
                # Add code if available
                if response.code:
                    html_content += f"""
                    <h2>Generated Code</h2>
                    <pre><code class="python">{response.code}</code></pre>
                    """
                
                # Add visualizations if available
                if response.visualizations:
                    html_content += "<h2>Visualizations</h2>"
                    
                    for i, viz in enumerate(response.visualizations):
                        html_content += f"""
                        <h3>Visualization {i+1}</h3>
                        <p>{viz}</p>
                        """
                
                html_content += """
                </body>
                </html>
                """
                
                # Export
                html_data = html_content.encode('utf-8')
                
                st.download_button(
                    label="Download HTML",
                    data=html_data,
                    file_name="analysis_results.html",
                    mime="text/html",
                    key="download_html"
                )