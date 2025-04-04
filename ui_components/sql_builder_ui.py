"""
SQL Builder UI Module for Multi-Source Julius AI Chatbot.

This module provides UI components for building SQL statements using Julius AI.
"""

import os
import streamlit as st
import pandas as pd
import re
from data_sources.sql_connector import SQLConnector
from julius_api import Julius

def render_sql_builder_sidebar():
    """
    Render the SQL builder UI in the sidebar.
    
    This function displays UI elements for:
    - Connecting to databases
    - Viewing database schema
    """
    st.markdown("### SQL Builder")
    
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
                
                # Button to show first 5 rows
                if st.button(f"Show First 5 Rows of {selected_table}"):
                    try:
                        sample_query = f"SELECT * FROM {selected_table} LIMIT 5"
                        success, sample_df, _ = sql_connector.execute_direct_sql(sample_query)
                        if success and sample_df is not None and not sample_df.empty:
                            st.dataframe(sample_df)
                        else:
                            st.warning("No data found or error executing query.")
                    except Exception as e:
                        st.error(f"Error getting sample data: {str(e)}")
    
    # Disconnect button (only show if connected)
    if st.session_state.get('sql_connected', False) and sql_connector.connected:
        if st.button("Disconnect"):
            sql_connector.disconnect()
            st.session_state['sql_connected'] = False
            st.success("Disconnected from database.")

def render_sql_builder_chat():
    """
    Render the SQL builder chat interface.
    
    This function displays:
    - Chat input for natural language query
    - Generated SQL statement
    - Option to execute the SQL statement
    """
    st.header("SQL Statement Builder")
    
    # Check if connected to database
    if not st.session_state.get('sql_connected', False):
        st.warning("Please connect to a database first using the sidebar.")
        return
    
    # Get SQL connector from session state
    sql_connector = st.session_state.get('sql_connector')
    if not sql_connector:
        st.error("SQL connector not initialized.")
        return
    
    # Get Julius client from session state
    julius = st.session_state.get('julius')
    if not julius:
        st.error("Julius client not initialized.")
        return
    
    # Get selected table from session state
    selected_table = st.session_state.get('selected_table')
    
    # Show which table is currently selected
    if selected_table:
        st.info(f"Currently focused on table: **{selected_table}**. Your SQL will be generated for this table.")
        
        # Option to clear table selection
        if st.button("Clear table selection"):
            st.session_state['selected_table'] = None
            st.experimental_rerun()
    else:
        st.info("No specific table selected. Your SQL will be generated for the entire database.")
    
    # Initialize SQL history in session state
    if 'sql_builder_history' not in st.session_state:
        st.session_state['sql_builder_history'] = []
    
    # Display SQL history
    if st.session_state['sql_builder_history']:
        with st.expander("SQL History", expanded=False):
            for i, (query, sql) in enumerate(st.session_state['sql_builder_history']):
                st.markdown(f"**Query {i+1}:** {query}")
                st.code(sql, language="sql")
                st.markdown("---")
    
    # Add a direct SQL execution option using a form
    st.markdown("---")
    st.markdown("### Direct SQL Execution")
    st.markdown("You can directly execute SQL queries without using the natural language generator.")
    
    # Use a form for direct SQL execution
    with st.form(key="direct_sql_form"):
        # Direct SQL input
        direct_sql = st.text_area("Enter SQL query directly", height=150)
        
        # Submit button
        direct_submit = st.form_submit_button("Execute SQL Query")
    
    # Handle form submission
    if direct_submit:
        if direct_sql:
            with st.spinner("Executing SQL query..."):
                try:
                    # Execute the SQL query
                    success, result_df, error_msg = sql_connector.execute_direct_sql(direct_sql)
                    
                    if success and result_df is not None:
                        # Store the result in session state
                        st.session_state['direct_sql_result'] = result_df
                        
                        # Display dataframe summary
                        st.markdown("#### Query Results Summary")
                        st.info(f"Rows: {result_df.shape[0]}, Columns: {result_df.shape[1]}")
                        
                        # Display the dataframe
                        st.markdown("#### Query Results")
                        st.dataframe(result_df)
                        
                        # Download options
                        st.markdown("#### Download Options")
                        download_col1, download_col2, download_col3 = st.columns(3)
                        
                        # CSV download
                        with download_col1:
                            csv = result_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="Download CSV",
                                data=csv,
                                file_name="direct_query_result.csv",
                                mime="text/csv"
                            )
                        
                        # Excel download
                        with download_col2:
                            import io
                            buffer = io.BytesIO()
                            result_df.to_excel(buffer, index=False, engine='openpyxl')
                            buffer.seek(0)
                            st.download_button(
                                label="Download Excel",
                                data=buffer,
                                file_name="direct_query_result.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                        
                        # JSON download
                        with download_col3:
                            json_str = result_df.to_json(orient='records')
                            st.download_button(
                                label="Download JSON",
                                data=json_str,
                                file_name="direct_query_result.json",
                                mime="application/json"
                            )
                        
                        # Option to save as dataset
                        with st.form(key="save_dataset_form"):
                            st.markdown("#### Save as Dataset")
                            dataset_name = st.text_input("Dataset Name", "direct_query_result.csv")
                            save_dataset = st.form_submit_button("Save as Dataset")
                            
                            if save_dataset:
                                # Initialize dataset handler if not already in session state
                                if 'dataset_handler' not in st.session_state:
                                    from data_sources.dataset_handler import DatasetHandler
                                    st.session_state['dataset_handler'] = DatasetHandler()
                                
                                # Get dataset handler from session state
                                dataset_handler = st.session_state['dataset_handler']
                                
                                # Get the result from session state
                                result_to_save = st.session_state['direct_sql_result']
                                
                                # Save as dataset
                                if dataset_handler.load_dataset_from_dataframe(result_to_save, dataset_name):
                                    # Update session state
                                    st.session_state['dataset'] = result_to_save
                                    st.session_state['dataset_name'] = dataset_name
                                    st.session_state['dataset_source'] = 'sql_builder'
                                    st.session_state['dataset_loaded'] = True
                                    st.success(f"Query results saved as dataset: {dataset_name}")
                                else:
                                    st.error("Failed to save query results as dataset.")
                    else:
                        st.error(f"Error executing SQL query: {error_msg}")
                except Exception as e:
                    st.error(f"Error executing SQL query: {str(e)}")
        else:
            st.warning("Please enter an SQL query.")
    
    # Display previous results if available
    if 'direct_sql_result' in st.session_state and not direct_submit:
        result_df = st.session_state['direct_sql_result']
        
        # Display dataframe summary
        st.markdown("#### Previous Query Results Summary")
        st.info(f"Rows: {result_df.shape[0]}, Columns: {result_df.shape[1]}")
        
        # Display the dataframe
        st.markdown("#### Previous Query Results")
        st.dataframe(result_df)
        
        # Download options
        st.markdown("#### Download Options")
        download_col1, download_col2, download_col3 = st.columns(3)
        
        # CSV download
        with download_col1:
            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="direct_query_result.csv",
                mime="text/csv"
            )
        
        # Excel download
        with download_col2:
            import io
            buffer = io.BytesIO()
            result_df.to_excel(buffer, index=False, engine='openpyxl')
            buffer.seek(0)
            st.download_button(
                label="Download Excel",
                data=buffer,
                file_name="direct_query_result.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        # JSON download
        with download_col3:
            json_str = result_df.to_json(orient='records')
            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name="direct_query_result.json",
                mime="application/json"
            )
    
    st.markdown("---")
    st.markdown("### Natural Language to SQL")
    st.markdown("Generate SQL from natural language queries.")
    
    # Use a form for natural language to SQL
    with st.form(key="nl_sql_form"):
        # Query input
        query = st.text_area("Enter your query in natural language",
                            "Example: Show me all customers who made a purchase in the last 30 days",
                            height=100)
        
        # Submit button
        nl_submit = st.form_submit_button("Generate SQL")
    
    # Handle form submission for natural language to SQL
    if nl_submit:
        if query and query != "Example: Show me all customers who made a purchase in the last 30 days":
            with st.spinner("Generating SQL..."):
                # Get database schema
                if selected_table:
                    # If a specific table is selected, focus the prompt on that table
                    table_info = sql_connector.get_table_info(selected_table)
                    
                    # Get sample data for the selected table
                    try:
                        sample_query = f"SELECT * FROM {selected_table} LIMIT 5"
                        success, sample_df, _ = sql_connector.execute_direct_sql(sample_query)
                        if success and sample_df is not None and not sample_df.empty:
                            sample_data = f"\nSample data:\n{sample_df.to_string()}\n\n"
                        else:
                            sample_data = "\n"
                    except Exception:
                        sample_data = "\n"
                    
                    prompt = f"""
                    I want you to generate an SQL query for a specific table in a database. Here's the table information:
                    
                    Table: {selected_table}
                    {table_info}
                    {sample_data}
                    
                    User Query: {query}
                    
                    Please generate ONLY:
                    1. A brief one-sentence interpretation of what the user is asking for
                    2. The appropriate SQL query to retrieve this information from the {selected_table} table
                    
                    Format your response exactly like this:
                    
                    Interpretation: [Your one-sentence interpretation]
                    
                    ```sql
                    [Your SQL query]
                    ```
                    
                    IMPORTANT GUIDELINES FOR SQL GENERATION:
                    - Make sure the SQL query is valid for the table schema provided
                    - ALWAYS use the exact table name: {selected_table}
                    - Do NOT include the word 'the' at the end of your SQL query
                    - Always use proper column names exactly as they appear in the schema
                    - Limit result sets to 1000 rows maximum to avoid performance issues
                    - Use proper SQL syntax for the MySQL database
                    - Include semicolons at the end of your queries
                    - DO NOT include any additional explanations, insights, or analysis
                    """
                else:
                    # If no specific table is selected, provide the full schema
                    schema_info = ""
                    tables = sql_connector.get_tables()
                    
                    if not tables:
                        st.error("No tables found in the database")
                        return
                    
                    # Build schema information
                    for table in tables:
                        # Get table info
                        table_info = sql_connector.get_table_info(table)
                        schema_info += f"Table: {table}\n{table_info}\n\n"
                    
                    prompt = f"""
                    I want you to generate an SQL query for a database. Here's the database schema information:
                    
                    {schema_info}
                    
                    User Query: {query}
                    
                    Please generate ONLY:
                    1. A brief one-sentence interpretation of what the user is asking for
                    2. The appropriate SQL query to retrieve this information
                    
                    Format your response exactly like this:
                    
                    Interpretation: [Your one-sentence interpretation]
                    
                    ```sql
                    [Your SQL query]
                    ```
                    
                    IMPORTANT GUIDELINES FOR SQL GENERATION:
                    - Make sure the SQL query is valid for the database schema provided
                    - Do NOT include the word 'the' at the end of your SQL query
                    - Always use proper table and column names exactly as they appear in the schema
                    - Limit result sets to 1000 rows maximum to avoid performance issues
                    - Use proper SQL syntax for the MySQL database
                    - Include semicolons at the end of your queries
                    - DO NOT include any additional explanations, insights, or analysis
                    """
                
                try:
                    # Send query to Julius
                    messages = [{"role": "user", "content": prompt}]
                    response = julius.chat.completions.create(messages)
                    
                    # Extract response content
                    response_content = response.message.content
                    
                    # Extract interpretation and SQL query
                    interpretation_pattern = r'Interpretation:\s*(.*?)(?:\n|$)'
                    interpretation_match = re.search(interpretation_pattern, response_content)
                    interpretation = interpretation_match.group(1).strip() if interpretation_match else "No interpretation provided"
                    
                    # Extract SQL query
                    sql_pattern = r'```sql\s*(.*?)\s*```'
                    sql_match = re.search(sql_pattern, response_content, re.DOTALL)
                    sql_query = sql_match.group(1).strip() if sql_match else "No SQL query generated"
                    
                    # Store in session state
                    st.session_state['nl_interpretation'] = interpretation
                    st.session_state['nl_sql_query'] = sql_query
                    
                    # Add to SQL history
                    st.session_state['sql_builder_history'].append((query, sql_query))
                    
                    # Display interpretation
                    st.markdown(f"**Interpretation:** {interpretation}")
                    
                    # Display SQL query
                    st.markdown("**Generated SQL:**")
                    st.code(sql_query, language="sql")
                    
                    # Create a form for executing the generated SQL
                    with st.form(key="execute_nl_sql_form"):
                        st.markdown("#### Execute Generated SQL")
                        execute_nl_sql = st.form_submit_button("Execute Generated SQL")
                        
                        if execute_nl_sql:
                            with st.spinner("Executing SQL query..."):
                                try:
                                    # Execute the SQL query
                                    success, result_df, error_msg = sql_connector.execute_direct_sql(sql_query)
                                    
                                    if success and result_df is not None:
                                        # Store the result in session state
                                        st.session_state['nl_sql_result'] = result_df
                                        
                                        # Display dataframe summary
                                        st.markdown("#### Query Results Summary")
                                        st.info(f"Rows: {result_df.shape[0]}, Columns: {result_df.shape[1]}")
                                        
                                        # Display the dataframe
                                        st.markdown("#### Query Results")
                                        st.dataframe(result_df)
                                        
                                        # Download options
                                        st.markdown("#### Download Options")
                                        download_col1, download_col2, download_col3 = st.columns(3)
                                        
                                        # CSV download
                                        with download_col1:
                                            csv = result_df.to_csv(index=False).encode('utf-8')
                                            st.download_button(
                                                label="Download CSV",
                                                data=csv,
                                                file_name="nl_query_result.csv",
                                                mime="text/csv"
                                            )
                                        
                                        # Excel download
                                        with download_col2:
                                            import io
                                            buffer = io.BytesIO()
                                            result_df.to_excel(buffer, index=False, engine='openpyxl')
                                            buffer.seek(0)
                                            st.download_button(
                                                label="Download Excel",
                                                data=buffer,
                                                file_name="nl_query_result.xlsx",
                                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                            )
                                        
                                        # JSON download
                                        with download_col3:
                                            json_str = result_df.to_json(orient='records')
                                            st.download_button(
                                                label="Download JSON",
                                                data=json_str,
                                                file_name="nl_query_result.json",
                                                mime="application/json"
                                            )
                                    else:
                                        st.error(f"Error executing SQL query: {error_msg}")
                                except Exception as e:
                                    st.error(f"Error executing SQL query: {str(e)}")
                except Exception as e:
                    st.error(f"Error generating SQL: {str(e)}")
                    
                    with st.spinner("Executing SQL query..."):
                        try:
                            success, result_df, error_msg = sql_connector.execute_direct_sql(current_sql_query)
                            
                            if success and result_df is not None:
                                # Display dataframe summary
                                st.markdown("#### Query Results Summary")
                                st.info(f"Rows: {result_df.shape[0]}, Columns: {result_df.shape[1]}")
                                
                                # Display the dataframe
                                st.markdown("#### Query Results")
                                st.dataframe(result_df)
                                
                                # Download options
                                st.markdown("#### Download Options")
                                download_col1, download_col2, download_col3 = st.columns(3)
                                
                                # CSV download
                                with download_col1:
                                    csv = result_df.to_csv(index=False).encode('utf-8')
                                    st.download_button(
                                        label="Download CSV",
                                        data=csv,
                                        file_name=f"query_result_{len(st.session_state['sql_builder_history'])}.csv",
                                        mime="text/csv",
                                        key=f"download_csv_{len(st.session_state['sql_builder_history'])}"
                                    )
                                
                                # Excel download
                                with download_col2:
                                    import io
                                    buffer = io.BytesIO()
                                    result_df.to_excel(buffer, index=False, engine='openpyxl')
                                    buffer.seek(0)
                                    st.download_button(
                                        label="Download Excel",
                                        data=buffer,
                                        file_name=f"query_result_{len(st.session_state['sql_builder_history'])}.xlsx",
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                        key=f"download_excel_{len(st.session_state['sql_builder_history'])}"
                                    )
                                
                                # JSON download
                                with download_col3:
                                    json_str = result_df.to_json(orient='records')
                                    st.download_button(
                                        label="Download JSON",
                                        data=json_str,
                                        file_name=f"query_result_{len(st.session_state['sql_builder_history'])}.json",
                                        mime="application/json",
                                        key=f"download_json_{len(st.session_state['sql_builder_history'])}"
                                    )
                                
                                # Option to save as dataset
                                st.markdown("#### Save as Dataset")
                                save_col1, save_col2 = st.columns([3, 1])
                                with save_col1:
                                    dataset_name = st.text_input("Dataset Name", f"sql_query_result_{len(st.session_state['sql_builder_history'])}.csv", key=f"dataset_name_{len(st.session_state['sql_builder_history'])}")
                                with save_col2:
                                    if st.button("Save as Dataset", key=f"save_dataset_{len(st.session_state['sql_builder_history'])}"):
                                        # Initialize dataset handler if not already in session state
                                        if 'dataset_handler' not in st.session_state:
                                            from data_sources.dataset_handler import DatasetHandler
                                            st.session_state['dataset_handler'] = DatasetHandler()
                                        
                                        # Get dataset handler from session state
                                        dataset_handler = st.session_state['dataset_handler']
                                        
                                        # Save as dataset
                                        if dataset_handler.load_dataset_from_dataframe(result_df, dataset_name):
                                            # Update session state
                                            st.session_state['dataset'] = result_df
                                            st.session_state['dataset_name'] = dataset_name
                                            st.session_state['dataset_source'] = 'sql_builder'
                                            st.session_state['dataset_loaded'] = True
                                            st.success(f"Query results saved as dataset: {dataset_name}")
                                        else:
                                            st.error("Failed to save query results as dataset.")
                            else:
                                st.error(f"Error executing SQL query: {error_msg}")
                        except Exception as e:
                            st.error(f"Error executing SQL query: {str(e)}")
        else:
            st.warning("Please enter a query.")