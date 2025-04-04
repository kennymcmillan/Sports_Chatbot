"""
Database Reasoning UI Module for Multi-Source Julius AI Chatbot.

This module provides UI components for direct database reasoning using Julius AI.
"""

import os
import streamlit as st
import pandas as pd
import re
import time
import requests
import logging
from io import BytesIO
from PIL import Image
from data_sources.db_reasoning import DatabaseReasoning
from data_sources.sql_connector import SQLConnector
from julius_api import Julius

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DBReasoningUI")
from data_sources.dataset_handler import DatasetHandler

def render_db_reasoning_sidebar():
    """
    Render the database reasoning UI in the sidebar.
    
    This function displays UI elements for:
    - Connecting to databases
    - Viewing database schema
    - Setting reasoning preferences
    """
    st.markdown("### Database Reasoning")
    
    # Initialize SQL connector if not already in session state
    if 'sql_connector' not in st.session_state:
        # Get OpenAI API key from environment
        import os
        openai_api_key = os.getenv("OPENAI_API_KEY")
        st.session_state['sql_connector'] = SQLConnector(api_key=openai_api_key)
    
    # Get SQL connector from session state
    sql_connector = st.session_state['sql_connector']
    
    # Initialize Julius client if not already in session state
    if 'julius' not in st.session_state:
        api_key_from_env = os.getenv("JULIUS_API_TOKEN")
        st.session_state['julius'] = Julius(api_key=api_key_from_env)
    
    # Get Julius client from session state
    julius = st.session_state['julius']
    
    # Initialize database reasoning if not already in session state
    if 'db_reasoning' not in st.session_state:
        st.session_state['db_reasoning'] = DatabaseReasoning(
            sql_connector=sql_connector,
            julius=julius
        )
    
    # Get database reasoning from session state
    db_reasoning = st.session_state['db_reasoning']
    
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
        if st.button("Connect to Database", key="db_reasoning_connect_env"):
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
    
    # Database schema section (only show if connected)
    if st.session_state.get('sql_connected', False) and sql_connector.connected:
        st.markdown("### Database Schema")
        
        # Display available tables
        tables = sql_connector.get_tables()
        if tables:
            # Table selection dropdown
            selected_table = st.selectbox("Select a table to view schema", tables)
            
            # Store selected table in session state
            st.session_state['selected_table'] = selected_table
            
            # Display table schema
            if selected_table:
                with st.expander("Table Schema", expanded=False):
                    schema_info = sql_connector.get_table_info(selected_table)
                    st.code(schema_info)
                
                # Option to view sample data
                if st.button(f"View Sample Data", key=f"view_sample_{selected_table}"):
                    try:
                        sample_query = f"SELECT * FROM {selected_table} LIMIT 10"
                        success, sample_df, _ = sql_connector.execute_direct_sql(sample_query)
                        if success and sample_df is not None:
                            st.dataframe(sample_df)
                    except Exception as e:
                        st.error(f"Error getting sample data: {str(e)}")
    
    # Disconnect button (only show if connected)
    if st.session_state.get('sql_connected', False) and sql_connector.connected:
        if st.button("Disconnect", key="db_reasoning_disconnect"):
            sql_connector.disconnect()
            st.session_state['sql_connected'] = False
            st.success("Disconnected from database.")

def render_db_reasoning_chat():
    """
    Render the database reasoning chat interface.
    
    This function displays:
    - Chat history
    - Chat input
    - Query results
    - Options to save query results as datasets
    """
    st.header("Database Reasoning Chat")
    
    # Check if connected to database
    if not st.session_state.get('sql_connected', False):
        st.warning("Please connect to a database first using the sidebar.")
        return
    
    # Get database reasoning from session state
    db_reasoning = st.session_state.get('db_reasoning')
    if not db_reasoning:
        st.error("Database reasoning module not initialized.")
        return
    
    # Get dataset handler from session state
    dataset_handler = st.session_state.get('dataset_handler')
    
    # Initialize chat history in session state
    if 'db_reasoning_messages' not in st.session_state:
        st.session_state['db_reasoning_messages'] = []
    
    # Initialize query results in session state
    if 'db_reasoning_results' not in st.session_state:
        st.session_state['db_reasoning_results'] = []
    
    # Create a scrollable container for chat with fixed height
    chat_container = st.container(height=400, border=True)
    
    # Display chat history
    with chat_container:
        # Clear any previous content
        st.empty()
        
        # Display all messages
        for i, message in enumerate(st.session_state['db_reasoning_messages']):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # If it's an assistant message with query results, display them
                if message["role"] == "assistant" and i < len(st.session_state['db_reasoning_results']):
                    result = st.session_state['db_reasoning_results'][i]
                    
                    # Display SQL query
                    if "sql_query" in result:
                        with st.expander("SQL Query", expanded=False):
                            st.code(result["sql_query"], language="sql")
                    
                    # Display query results
                    if "results" in result and isinstance(result["results"], pd.DataFrame):
                        st.dataframe(result["results"])
                        
                        # Option to save as dataset
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            dataset_name = st.text_input(
                                "Dataset Name", 
                                f"db_query_result_{i}.csv",
                                key=f"dataset_name_{i}"
                            )
                        with col2:
                            if st.button("Save as Dataset", key=f"save_dataset_{i}"):
                                # Save as dataset
                                if dataset_handler.load_dataset_from_dataframe(
                                    result["results"], 
                                    dataset_name
                                ):
                                    # Update session state
                                    st.session_state['dataset'] = result["results"]
                                    st.session_state['dataset_name'] = dataset_name
                                    st.session_state['dataset_source'] = 'db_reasoning'
                                    st.session_state['dataset_loaded'] = True
                                    st.success(f"Query results saved as dataset: {dataset_name}")
                                else:
                                    st.error("Failed to save query results as dataset.")
    
    # Get selected table from session state
    selected_table = st.session_state.get('selected_table')
    
    # Show which table is currently selected
    if selected_table:
        st.info(f"Currently focused on table: **{selected_table}**. Your questions will be directed to this table.")
        
        # Option to clear table selection
        if st.button("Clear table selection", key="clear_table_selection"):
            st.session_state['selected_table'] = None
            st.experimental_rerun()
    else:
        st.info("No specific table selected. Your questions will be directed to the entire database.")
    
    # Code generation option
    generate_code = st.checkbox("Generate Python code for data analysis", value=False)
    
    # Chat input
    prompt = st.chat_input("Ask a question about your database...")
    
    if prompt:
        # Add user message to chat history
        st.session_state['db_reasoning_messages'].append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Query the database
        with st.spinner("Analyzing database and generating response..." + (" and code" if generate_code else "")):
            # Start timing
            start_time = time.time()
            logger.info(f"Starting database reasoning for query: {prompt[:50]}...")
            
            # Pass the selected table and code generation flag to the query_database method
            logger.info(f"Querying database with table: {selected_table}")
            query_start = time.time()
            success, response = db_reasoning.query_database(prompt, selected_table, generate_code)
            query_time = time.time() - query_start
            logger.info(f"Database query completed in {query_time:.2f} seconds")
            
            if success:
                # Add assistant message to chat history
                st.session_state['db_reasoning_messages'].append({
                    "role": "assistant",
                    "content": response["response"]
                })
                
                # Add query results to results history
                st.session_state['db_reasoning_results'].append(response)
                
                # Log success
                if "sql_query" in response:
                    logger.info(f"Generated SQL query: {response['sql_query'][:100]}...")
                
                if "results" in response and isinstance(response["results"], pd.DataFrame):
                    logger.info(f"Query returned {len(response['results'])} rows")
                
                # Display assistant message
                with st.chat_message("assistant"):
                    st.markdown(response["response"])
                    
                    # Display SQL query
                    if "sql_query" in response:
                        with st.expander("SQL Query", expanded=False):
                            st.code(response["sql_query"], language="sql")
                    
                    # Process and display images from Julius response
                    if "response" in response:
                        logger.info("Processing images from response...")
                        image_start = time.time()
                        
                        # Extract image URLs from the response text
                        image_pattern = r'\{\{images\[(\d+)\]\}\}'
                        image_matches = re.findall(image_pattern, response["response"])
                        
                        # Extract image URLs from the JSON output
                        image_urls = []
                        if "image_urls" in response["response"]:
                            try:
                                # Try to extract from JSON-like structure
                                url_pattern = r'"image_urls":\s*\[(.*?)\]'
                                url_matches = re.findall(url_pattern, response["response"], re.DOTALL)
                                if url_matches:
                                    # Extract URLs from the matched string
                                    url_list = url_matches[0]
                                    url_pattern = r'"(https?://[^"]+)"'
                                    image_urls = re.findall(url_pattern, url_list)
                                    logger.info(f"Found {len(image_urls)} image URLs in JSON structure")
                            except Exception as e:
                                logger.warning(f"Error extracting image URLs: {str(e)}")
                        
                        # Display images if found
                        if image_urls:
                            st.subheader("Generated Visualizations")
                            saved_images = 0
                            for i, url in enumerate(image_urls):
                                try:
                                    # Download and display the image
                                    response_img = requests.get(url)
                                    if response_img.status_code == 200:
                                        img = Image.open(BytesIO(response_img.content))
                                        st.image(img, caption=f"Visualization {i+1}")
                                        
                                        # Save the image to the outputs directory
                                        os.makedirs("outputs", exist_ok=True)
                                        img_path = f"outputs/visualization_{int(time.time())}_{i}.png"
                                        img.save(img_path)
                                        saved_images += 1
                                        logger.info(f"Saved image to {img_path}")
                                except Exception as e:
                                    logger.error(f"Error displaying image {i+1}: {str(e)}")
                            
                            logger.info(f"Processed and saved {saved_images} images in {time.time() - image_start:.2f} seconds")
                    
                    # Display generated code if available
                    if "code" in response and response["code"]:
                        logger.info(f"Processing generated code ({len(response['code'])} bytes)")
                        with st.expander("Generated Python Code", expanded=False):
                            st.code(response["code"], language="python")
                            
                            # Option to download code
                            if "code_filename" in response and response["code_filename"]:
                                with open(response["code_filename"], "r") as f:
                                    code_content = f.read()
                                    
                                st.download_button(
                                    label="Download Code",
                                    data=code_content,
                                    file_name=os.path.basename(response["code_filename"]),
                                    mime="text/plain"
                                )
                            
                            # Option to execute code
                            if st.button("Execute Code", key=f"execute_code_{len(st.session_state['db_reasoning_results'])}"):
                                logger.info("Executing generated code...")
                                code_start = time.time()
                                with st.spinner("Executing code..."):
                                    try:
                                        # Save the data to a temporary CSV file
                                        if "results" in response and isinstance(response["results"], pd.DataFrame):
                                            temp_csv = f"outputs/temp_data_{int(time.time())}.csv"
                                            response["results"].to_csv(temp_csv, index=False)
                                            
                                            # Modify the code to use the temporary CSV file
                                            modified_code = response["code"].replace(
                                                "pd.read_csv('",
                                                f"pd.read_csv('{temp_csv}"
                                            )
                                            
                                            # Create a temporary Python file
                                            temp_py = f"outputs/temp_script_{int(time.time())}.py"
                                            with open(temp_py, "w") as f:
                                                f.write(modified_code)
                                            
                                            # Execute the code and capture output
                                            import subprocess
                                            import sys
                                            
                                            result = subprocess.run(
                                                [sys.executable, temp_py],
                                                capture_output=True,
                                                text=True
                                            )
                                            
                                            # Display output
                                            if result.stdout:
                                                st.subheader("Code Output:")
                                                st.text(result.stdout)
                                            
                                            # Display errors
                                            if result.stderr:
                                                st.error("Code Execution Error:")
                                                st.code(result.stderr)
                                            
                                            # Check for generated images
                                            import glob
                                            image_files = glob.glob("*.png") + glob.glob("*.jpg")
                                            
                                            if image_files:
                                                logger.info(f"Found {len(image_files)} generated image files")
                                                st.subheader("Generated Visualizations:")
                                                for img_file in image_files:
                                                    st.image(img_file)
                                    except Exception as e:
                                        logger.error(f"Error executing code: {str(e)}")
                                        st.error(f"Error executing code: {str(e)}")
                                
                                logger.info(f"Code execution completed in {time.time() - code_start:.2f} seconds")
                    
                    # Display query results
                    if "results" in response and isinstance(response["results"], pd.DataFrame):
                        st.dataframe(response["results"])
                        
                        # Option to save as dataset
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            dataset_name = st.text_input(
                                "Dataset Name", 
                                f"db_query_result_{len(st.session_state['db_reasoning_results'])}.csv"
                            )
                        with col2:
                            if st.button("Save as Dataset", key=f"save_dataset_current_{len(st.session_state['db_reasoning_results'])}"):
                                # Save as dataset
                                if dataset_handler.load_dataset_from_dataframe(
                                    response["results"], 
                                    dataset_name
                                ):
                                    # Update session state
                                    st.session_state['dataset'] = response["results"]
                                    st.session_state['dataset_name'] = dataset_name
                                    st.session_state['dataset_source'] = 'db_reasoning'
                                    st.session_state['dataset_loaded'] = True
                                    st.success(f"Query results saved as dataset: {dataset_name}")
                                else:
                                    st.error("Failed to save query results as dataset.")
                
                # Log total time
                total_time = time.time() - start_time
                logger.info(f"Total database reasoning operation completed in {total_time:.2f} seconds")
            
            else:
                # Display error message
                error_message = response.get("error", "Unknown error occurred")
                logger.error(f"Database reasoning failed: {error_message}")
                st.error(f"Error: {error_message}")
                
                # Add error message to chat history
                st.session_state['db_reasoning_messages'].append({
                    "role": "assistant", 
                    "content": f"I encountered an error: {error_message}"
                })
                
                # Add empty result to results history to maintain alignment
                st.session_state['db_reasoning_results'].append({})