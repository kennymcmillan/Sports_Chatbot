"""
Multi-Source Julius AI Chatbot Application

This application provides a streamlined interface for:
1. Loading and analyzing data from various sources
2. Querying databases using natural language
3. Generating insights and visualizations

The application uses a service-based architecture with:
- Core services for data, database, AI, and export functionality
- UI components for data exploration, query building, and analysis
"""

import os
import sys
import time
import pandas as pd
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv

# IMPORTANT: This must be the first Streamlit command
st.set_page_config(
    page_title="Aspire Academy Sports Analytics",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Aspire Academy Sports Analytics Platform",
        'Get Help': 'https://aspire.qa/contact-us',
        'Report a bug': "https://aspire.qa/contact-us"
    }
)

# Load environment variables
load_dotenv()

# Import core services
from core_services.data_service import DataService
from core_services.database_service import DatabaseService
from core_services.ai_service import AIService, AIRequest
from core_services.export_service import ExportService

# Import UI components
from ui_components.data_explorer import render_data_explorer_sidebar, render_data_explorer
from ui_components.query_builder import render_query_builder_sidebar, render_query_builder
from ui_components.analysis_dashboard import render_analysis_dashboard_sidebar, render_analysis_dashboard
from ui_components.database_reasoning_ui import render_database_reasoning_ui, render_database_reasoning_sidebar
from ui_components.database_reasoning_adapter import render_database_reasoning_ui_adapter

def set_custom_style():
    """
    Set custom styling to match Aspire Academy's brand colors.
    """
    st.markdown("""
        <style>
        /* Main background and text colors */
        .stApp {
            background-color: #FFFFFF;
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #1B365D;
            color: #FFFFFF;
        }
        
        [data-testid="stSidebar"] .stMarkdown {
            color: #FFFFFF;
        }
        
        /* Headers */
        h1, h2, h3 {
            color: #1B365D;
        }
        
        /* Buttons */
        .stButton button {
            background-color: #00A3E0;
            color: #FFFFFF;
            border: none;
        }
        
        .stButton button:hover {
            background-color: #1B365D;
            color: #FFFFFF;
        }
        
        /* Radio buttons and checkboxes */
        .stRadio label, .stCheckbox label {
            color: #1B365D;
        }
        
        /* Success messages */
        .element-container .stAlert.success {
            background-color: #00A3E0;
            color: #FFFFFF;
        }
        
        /* Info messages */
        .element-container .stAlert.info {
            background-color: #F5F5F5;
            border-left-color: #00A3E0;
        }
        
        /* Metrics */
        [data-testid="stMetricValue"] {
            color: #1B365D;
        }
        
        /* DataFrames */
        .stDataFrame {
            border: 1px solid #1B365D;
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] button[data-baseweb="tab"] {
            color: #1B365D;
        }
        
        .stTabs [data-baseweb="tab-list"] button[data-baseweb="tab"]:hover {
            color: #00A3E0;
        }
        
        /* Selectbox */
        .stSelectbox label {
            color: #1B365D;
        }
        
        /* Links */
        a {
            color: #00A3E0;
        }
        
        a:hover {
            color: #1B365D;
        }
        </style>
    """, unsafe_allow_html=True)

# Service initialization functions with error handling
def get_data_service():
    """
    Initialize or retrieve the DataService from the session state.
    
    The DataService handles:
    - File uploads (CSV, Excel, Parquet, etc.)
    - Data preprocessing and validation
    - Basic data analysis functions
    
    Returns:
        DataService: The initialized data service instance
        None: If initialization fails
    """
    if 'data_service' not in st.session_state:
        try:
            st.session_state['data_service'] = DataService()
        except Exception as e:
            st.error(f"Failed to initialize Data Service: {str(e)}")
            return None
    return st.session_state['data_service']

def get_database_service():
    """
    Initialize or retrieve the DatabaseService from the session state.
    
    The DatabaseService handles:
    - Database connections (MySQL, PostgreSQL, SQLite)
    - Query execution and validation
    - Schema management and exploration
    
    Returns:
        DatabaseService: The initialized database service instance
        None: If initialization fails
    """
    if 'database_service' not in st.session_state:
        try:
            database_service = DatabaseService()
            st.session_state['database_service'] = database_service
            # Initialize connection state
            if 'db_connected' not in st.session_state:
                st.session_state['db_connected'] = False
            if 'current_connection' not in st.session_state:
                st.session_state['current_connection'] = None
        except Exception as e:
            st.error(f"Failed to initialize Database Service: {str(e)}")
            return None
    return st.session_state['database_service']

def get_ai_service():
    """
    Initialize or retrieve the AIService from the session state.
    
    The AIService handles:
    - Integration with Julius AI
    - Natural language processing
    - Query generation and analysis
    - Data visualization suggestions
    
    Returns:
        AIService: The initialized AI service instance
        None: If initialization fails or API key is missing
    """
    if 'ai_service' not in st.session_state:
        api_key = os.getenv("JULIUS_API_TOKEN")
        if not api_key:
            st.error("Julius API key not found. Please set the JULIUS_API_TOKEN environment variable.")
            return None
        try:
            st.session_state['ai_service'] = AIService(api_key)
        except Exception as e:
            st.error(f"Failed to initialize AI Service: {str(e)}")
            return None
    return st.session_state['ai_service']

def get_export_service():
    """
    Initialize or retrieve the ExportService from the session state.
    
    The ExportService handles:
    - Data export to various formats
    - Report generation
    - Visualization export
    
    Returns:
        ExportService: The initialized export service instance
        None: If initialization fails
    """
    if 'export_service' not in st.session_state:
        try:
            st.session_state['export_service'] = ExportService()
        except Exception as e:
            st.error(f"Failed to initialize Export Service: {str(e)}")
            return None
    return st.session_state['export_service']

def init_session_state():
    """
    Initialize all required session state variables.
    
    Sets up default values for:
    - Data source selection
    - Dataset loading status
    - Database connection status
    - Current analysis mode
    - History tracking (analysis and chat)
    - Database reasoning settings
    - UI state preferences
    """
    if 'data_source' not in st.session_state:
        st.session_state.data_source = None  # 'file' or 'database'
    if 'dataset_loaded' not in st.session_state:
        st.session_state.dataset_loaded = False
    if 'db_connected' not in st.session_state:
        st.session_state.db_connected = False
    if 'current_connection' not in st.session_state:
        st.session_state.current_connection = None
    if 'current_mode' not in st.session_state:
        st.session_state.current_mode = None
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'selected_table' not in st.session_state:
        st.session_state.selected_table = None
    # Database reasoning specific variables
    if 'db_reasoning_chat_history' not in st.session_state:
        st.session_state.db_reasoning_chat_history = []
    if 'db_reasoning_mode' not in st.session_state:
        st.session_state.db_reasoning_mode = "Simple"  # "Simple" or "Advanced"
    if 'db_reasoning_show_schema' not in st.session_state:
        st.session_state.db_reasoning_show_schema = False
    if 'db_reasoning_show_code' not in st.session_state:
        st.session_state.db_reasoning_show_code = False
    if 'db_reasoning_show_visualizations' not in st.session_state:
        st.session_state.db_reasoning_show_visualizations = False

def main():
    """
    Main application entry point.
    
    Flow:
    1. Initialize sidebar and services
    2. Handle data source selection (file/database)
    3. Manage database connections if needed
    4. Present analysis mode selection
    5. Render appropriate UI components based on mode
    
    The application follows a modular structure where each mode
    (Data Explorer, Query Builder, Database Reasoning) has its own
    UI components and functionality while sharing core services.
    """
    # Apply custom styling
    set_custom_style()
    
    # Initialize session state
    init_session_state()
    
    # Get services using lazy loading
    data_service = get_data_service()
    database_service = get_database_service()
    ai_service = get_ai_service()
    export_service = get_export_service()
    
    # Check if services are available
    if not all([data_service, database_service, ai_service, export_service]):
        st.error("Some services failed to initialize. Please check the error messages above.")
        return

    # Sidebar for data source selection and configuration
    with st.sidebar:
        st.header("Data Source")
        
        # Data source selection
        data_source = st.radio(
            "Select Data Source",
            ["File Upload", "Database Connection"],
            key="data_source_selector"
        )
        
        # Handle File Upload
        if data_source == "File Upload":
            st.session_state.data_source = 'file'
            uploaded_file = st.file_uploader(
                "Upload Dataset", 
                type=["csv", "xlsx", "xls", "parquet", "json", "arrow", "avro", "txt"]
            )
            
            if uploaded_file:
                try:
                    success, df, error = data_service.load_file(uploaded_file)
                    if success:
                        st.session_state.dataset_loaded = True
                        st.session_state.current_dataset = df
                        st.session_state.dataset_name = uploaded_file.name
                        st.success(f"Dataset loaded: {uploaded_file.name}")
                        
                        # Display dataset info
                        st.markdown("### Dataset Information")
                        st.write(f"Rows: {df.height}")
                        st.write(f"Columns: {df.width}")
                        st.write(f"Memory Usage: {df.estimated_size() / (1024 * 1024):.2f} MB")
                        
                        # Display sample data
                        st.markdown("### Sample Data")
                        st.dataframe(df.head())
                    else:
                        st.error(f"Error loading dataset: {error}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        # Handle Database Connection
        elif data_source == "Database Connection":
            st.session_state.data_source = 'database'
            
            # Database connection form
            with st.form("db_connection_form"):
                use_env = st.checkbox("Use environment variables", value=True)
                
                if not use_env:
                    db_type = st.selectbox("Database Type", ["MySQL", "PostgreSQL", "SQLite"])
                    host = st.text_input("Host", "localhost")
                    port = st.number_input("Port", value=3306)
                    database = st.text_input("Database Name")
                    user = st.text_input("Username")
                    password = st.text_input("Password", type="password")
                
                connect_button = st.form_submit_button("Connect")
                
                if connect_button:
                    with st.spinner("Connecting to database..."):
                        if use_env:
                            success, error = database_service.connect(
                                db_type="mysql",  # Default to mysql when using env vars
                                use_env=True
                            )
                        else:
                            success, error = database_service.connect(
                                db_type=db_type.lower(),
                                host=host,
                                port=port,
                                database=database,
                                user=user,
                                password=password
                            )
                        
                        if success:
                            st.session_state.db_connected = True
                            st.success("Connected to database successfully!")
                        else:
                            st.error(f"Failed to connect: {error}")
            
            # Show table selection if connected
            if st.session_state.db_connected:
                tables = database_service.get_tables()
                if tables:
                    selected_table = st.selectbox(
                        "Select Table",
                        tables,
                        key="table_selector"
                    )
                    
                    if selected_table:
                        st.session_state.selected_table = selected_table
                        # Load sample data
                        success, df, error = database_service.execute_query(
                            f"SELECT * FROM {selected_table} LIMIT 5"
                        )
                        if success:
                            st.dataframe(df.to_pandas())
                else:
                    st.warning("No tables found in the database.")

        # Only show mode selection if data is loaded or database is connected
        if st.session_state.dataset_loaded or (st.session_state.db_connected and st.session_state.selected_table):
            st.markdown("---")
            st.header("Analysis Mode")
            
            modes = [
                "Data Explorer",
                "Query Builder",
                "Database Reasoning"
            ]
            
            selected_mode = st.radio(
                "Select Mode",
                modes,
                key="mode_selector"
            )
            
            if selected_mode != st.session_state.current_mode:
                st.session_state.current_mode = selected_mode
                st.rerun()

            # Add reasoning complexity selection for Database Reasoning mode
            if selected_mode == "Database Reasoning":
                st.markdown("---")
                st.header("Reasoning Complexity")
                complexity = st.radio(
                    "Select Complexity Level",
                    ["Simple", "Advanced"],
                    key="reasoning_complexity",
                    help="Simple mode focuses on direct SQL queries and results. Advanced mode includes external context, visualizations, and deeper analysis."
                )
                st.session_state.reasoning_mode = complexity.lower()

    # Main content area
    if not st.session_state.dataset_loaded and not st.session_state.db_connected:
        st.info("Please select a data source and load data or connect to a database.")
        return

    if not st.session_state.current_mode:
        st.info("Please select an analysis mode from the sidebar.")
        return

    # Render mode-specific content and sidebar
    if st.session_state.current_mode == "Data Explorer":
        with st.sidebar:
            if st.session_state['current_mode'] == 'data_explorer':
                pass  # File upload is already in the common sidebar
        render_data_explorer(data_service, ai_service)
    elif st.session_state.current_mode == "Query Builder":
        with st.sidebar:
            if st.session_state.db_connected:
                render_query_builder_sidebar(database_service)
        if st.session_state.db_connected and st.session_state.selected_table:
            render_query_builder(database_service, ai_service, data_service, export_service)
        else:
            st.warning("Please connect to a database and select a table first.")
    elif st.session_state.current_mode == "Database Reasoning":
        with st.sidebar:
            if st.session_state.db_connected:
                render_database_reasoning_sidebar(database_service)
        if st.session_state.db_connected and st.session_state.selected_table:
            # Pass only the required services to the updated UI function
            render_database_reasoning_ui(database_service, ai_service, data_service, export_service)
        else:
            st.warning("Please connect to a database and select a table first.")

if __name__ == "__main__":
    main()