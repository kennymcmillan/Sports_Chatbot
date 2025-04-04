"""
SQL UI Module for Multi-Source Julius AI Chatbot.

This module provides UI components for SQL database connection and querying.
"""

import streamlit as st
import pandas as pd
from data_sources.sql_connector import SQLConnector
from data_sources.dataset_handler import DatasetHandler

def render_sql_sidebar():
    """
    Render the SQL database connection UI in the sidebar.
    
    This function displays UI elements for:
    - Connecting to SQL databases
    - Executing natural language queries
    - Viewing query results
    - Loading query results as datasets
    """
    st.markdown("### SQL Database Connection")
    
    # Initialize SQL connector if not already in session state
    if 'sql_connector' not in st.session_state:
        # Get OpenAI API key from environment
        import os
        openai_api_key = os.getenv("OPENAI_API_KEY")
        st.session_state['sql_connector'] = SQLConnector(api_key=openai_api_key)
    
    # Get SQL connector from session state
    sql_connector = st.session_state['sql_connector']
    
    # Initialize dataset handler if not already in session state
    if 'dataset_handler' not in st.session_state:
        st.session_state['dataset_handler'] = DatasetHandler()
    
    # Get dataset handler from session state
    dataset_handler = st.session_state['dataset_handler']
    
    # Database connection options
    st.markdown("#### Connect to Database")
    
    # Option to use environment variables
    use_env_vars = st.checkbox("Use environment variables for connection", value=True)
    
    if use_env_vars:
        # Connect using environment variables
        if st.button("Connect to Database"):
            if sql_connector.connect_to_database(use_env=True):
                st.session_state['sql_connected'] = True
                st.success(f"Connected to database using environment variables")
            else:
                st.session_state['sql_connected'] = False
                st.error("Failed to connect to database using environment variables.")
    else:
        # Manual database connection form
        with st.form("db_connection_form"):
            db_type = st.selectbox("Database Type", ["MySQL", "PostgreSQL", "SQLite"])
            
            # SQLite only needs a database file path
            if db_type == "SQLite":
                database = st.text_input("Database File Path", "database.db")
                host = ""
                port = 0
                user = ""
                password = ""
            else:
                host = st.text_input("Host", "localhost")
                port = st.number_input("Port", value=3306 if db_type == "MySQL" else 5432)
                database = st.text_input("Database Name")
                user = st.text_input("Username")
                password = st.text_input("Password", type="password")
            
            submit_button = st.form_submit_button("Connect")
            
            if submit_button:
                if sql_connector.connect_to_database(
                    db_type=db_type.lower(),
                    host=host,
                    user=user,
                    password=password,
                    database=database,
                    port=port
                ):
                    st.session_state['sql_connected'] = True
                    st.success(f"Connected to {db_type} database: {database}")
                else:
                    st.session_state['sql_connected'] = False
                    st.error("Failed to connect to database.")
    
    # SQL query section (only show if connected)
    if st.session_state.get('sql_connected', False) and sql_connector.connected:
        st.markdown("### SQL Query")
        
        # Display available tables
        tables = sql_connector.get_tables()
        if tables:
            # Table selection dropdown
            selected_table = st.selectbox("Select a table to query", tables)
            
            # Display table schema
            if selected_table:
                with st.expander("Table Schema", expanded=False):
                    schema_info = sql_connector.get_table_info(selected_table)
                    st.code(schema_info)
                
                # Quick query buttons
                st.markdown("#### Quick Queries")
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button(f"View all rows"):
                        query = f"Show me all rows from the {selected_table} table"
                        st.session_state['nl_query'] = query
                with col2:
                    if st.button(f"View first 10 rows"):
                        query = f"Show me the first 10 rows from the {selected_table} table"
                        st.session_state['nl_query'] = query
                with col3:
                    if st.button(f"Count rows"):
                        query = f"Count the number of rows in the {selected_table} table"
                        st.session_state['nl_query'] = query
            
            # All tables expander
            with st.expander("All Available Tables", expanded=False):
                for table in tables:
                    st.write(f"- {table}")
                    # Option to view table schema
                    if st.button(f"View Schema: {table}", key=f"schema_{table}"):
                        schema_info = sql_connector.get_table_info(table)
                        st.code(schema_info)
        
        # Query input tabs
        query_tab1, query_tab2 = st.tabs(["Natural Language Query", "Direct SQL Query"])
        
        with query_tab1:
            # Natural language query input
            nl_query = st.text_area("Enter your query in natural language",
                                   st.session_state.get('nl_query', f"Show me the first 10 rows from the {selected_table if tables else 'customers'} table"))
            
            if st.button("Execute Natural Language Query"):
                if nl_query:
                    with st.spinner("Executing query..."):
                        # Execute the query
                        success, result_df, error_msg = sql_connector.execute_query(nl_query)
                        
                        if success and result_df is not None:
                            # Display the results
                            st.markdown("#### Query Results")
                            st.dataframe(result_df)
                            
                            # Option to load as dataset
                            if st.button("Load as Dataset", key="load_nl_dataset"):
                                # Generate a name for the dataset
                                import re
                                import datetime
                                # Create a safe filename from the query
                                safe_name = re.sub(r'[^\w\s]', '', nl_query[:30])
                                safe_name = re.sub(r'\s+', '_', safe_name).lower()
                                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                                dataset_name = f"sql_{safe_name}_{timestamp}.csv"
                                
                                # Load the dataset
                                if dataset_handler.load_dataset_from_sql(result_df, dataset_name):
                                    # Get dataset from handler
                                    df, name, info = dataset_handler.get_dataset_for_julius()
                                    
                                    # Update session state
                                    st.session_state['dataset'] = df
                                    st.session_state['dataset_name'] = name
                                    st.session_state['dataset_source'] = 'sql'
                                    st.session_state['using_processed_dataset'] = False
                                    
                                    # Switch to dataset mode
                                    st.session_state['current_mode'] = 'dataset'
                                    st.success(f"Query results loaded as dataset: {name}")
                                    st.info("Switched to Dataset mode for analysis.")
                                else:
                                    st.error("Failed to load query results as dataset.")
                        else:
                            st.error(f"Query execution failed: {error_msg}")
                else:
                    st.warning("Please enter a query.")
        
        with query_tab2:
            # Direct SQL query input
            sql_query = st.text_area("Enter your SQL query",
                                   st.session_state.get('sql_query', f"SELECT * FROM {selected_table if tables else 'customers'} LIMIT 10"))
            
            if st.button("Execute SQL Query"):
                if sql_query:
                    with st.spinner("Executing SQL query..."):
                        # Execute the query directly
                        success, result_df, error_msg = sql_connector.execute_direct_sql(sql_query)
                        
                        if success and result_df is not None:
                            # Display the results
                            st.markdown("#### Query Results")
                            st.dataframe(result_df)
                            
                            # Option to load as dataset
                            if st.button("Load as Dataset", key="load_sql_dataset"):
                                # Generate a name for the dataset
                                import re
                                import datetime
                                # Create a safe filename from the query
                                safe_name = re.sub(r'[^\w\s]', '', sql_query[:30])
                                safe_name = re.sub(r'\s+', '_', safe_name).lower()
                                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                                dataset_name = f"sql_{safe_name}_{timestamp}.csv"
                                
                                # Load the dataset
                                if dataset_handler.load_dataset_from_sql(result_df, dataset_name):
                                    # Get dataset from handler
                                    df, name, info = dataset_handler.get_dataset_for_julius()
                                    
                                    # Update session state
                                    st.session_state['dataset'] = df
                                    st.session_state['dataset_name'] = name
                                    st.session_state['dataset_source'] = 'sql'
                                    st.session_state['using_processed_dataset'] = False
                                    
                                    # Switch to dataset mode
                                    st.session_state['current_mode'] = 'dataset'
                                    st.success(f"Query results loaded as dataset: {name}")
                                    st.info("Switched to Dataset mode for analysis.")
                                else:
                                    st.error("Failed to load query results as dataset.")
                        else:
                            st.error(f"Query execution failed: {error_msg}")
                else:
                    st.warning("Please enter a SQL query.")
    
    # Disconnect button (only show if connected)
    if st.session_state.get('sql_connected', False) and sql_connector.connected:
        if st.button("Disconnect"):
            sql_connector.disconnect()
            st.session_state['sql_connected'] = False
            st.success("Disconnected from database.")