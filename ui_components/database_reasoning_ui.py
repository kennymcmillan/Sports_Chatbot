"""
Database Reasoning UI Module for Multi-Source Julius AI Chatbot.

This module provides UI components for analyzing databases using Julius AI.
"""

import os
import streamlit as st
import pandas as pd
import polars as pl
import time
import re
from typing import Optional, Dict, List, Any, Tuple
import json
import requests
from io import BytesIO
from PIL import Image
from sqlalchemy import text

from core_services.database_service import DatabaseService, TableSchema
from core_services.ai_service import AIService, DatabaseReasoningRequest, DatabaseReasoningResponse
from core_services.data_service import DataService
from core_services.export_service import ExportService
from custom_display import display_database_reasoning_response
# Removed: from julius_api import Julius
# Removed: from data_sources.db_reasoning import DatabaseReasoning

def render_database_reasoning_sidebar(database_service: DatabaseService):
    """
    Render the database reasoning sidebar.
    
    Args:
        database_service: DatabaseService instance
    """
    st.markdown("### Database Reasoning")
    
    # Display current mode from session state
    current_mode = st.session_state.get('reasoning_mode', 'simple')
    st.markdown(f"#### Current Mode: **{current_mode.capitalize()}**")
    
    # Show description based on current mode
    if current_mode == "simple":
        st.info("Simple mode focuses on direct SQL queries and results.")
    else:
        st.info("Advanced mode includes external context, visualizations, and deeper analysis.")
    
    # Check if connected to database and table selected
    if not database_service.current_connection or not st.session_state.get('selected_table'):
        st.info("Connect to a database and select a table in the 'Data Source' section first.")
        return

    selected_table = st.session_state.get('selected_table')
    current_connection = database_service.current_connection

    st.markdown("#### Selected Table Schema")
    st.write(f"**Table:** `{selected_table}`")

    # Display table schema if a specific table is selected
    if selected_table:
        schema = database_service.get_table_schema(selected_table, current_connection)
        if schema:
            with st.expander("Show Schema Details", expanded=False):
                # Display columns
                st.markdown("##### Columns")
                for col in schema.columns:
                    nullable = "NULL" if col.get('nullable', True) else "NOT NULL"
                    default = f"DEFAULT {col.get('default')}" if col.get('default') is not None else ""
                    st.code(f"{col['name']} {col['type']} {nullable} {default}")

                # Display primary keys
                if schema.primary_keys:
                    st.markdown("##### Primary Keys")
                    st.code(", ".join(schema.primary_keys))

                # Display foreign keys
                if schema.foreign_keys:
                    st.markdown("##### Foreign Keys")
                    for fk in schema.foreign_keys:
                        st.code(f"{', '.join(fk['constrained_columns'])} -> {fk['referred_table']}({', '.join(fk['referred_columns'])})")

            # Button to show sample data
            if st.button(f"Show Sample Data for {selected_table}", key="db_reasoning_show_sample"):
                with st.spinner("Loading sample data..."):
                    success, result, error = database_service.execute_query(
                        f"SELECT * FROM {selected_table} LIMIT 10",
                        connection_name=current_connection
                    )

                    if success and result is not None:
                        st.dataframe(result.to_pandas())
                    else:
                        st.error(f"Error loading sample data: {error}")
        else:
             st.warning(f"Could not retrieve schema for table: {selected_table}")

    # Analysis options
    st.markdown("#### Analysis Options")

    # Option to generate code (linked to reasoning_mode)
    generate_code_checked = st.session_state['reasoning_mode'] == 'advanced'
    st.checkbox("Generate Python code (Advanced Mode)", value=generate_code_checked, key="db_generate_code", disabled=True)

def get_important_columns(df: pd.DataFrame) -> List[str]:
    """
    Identify important columns in the DataFrame based on common patterns.
    
    Args:
        df: Input DataFrame
        
    Returns:
        List of important column names
    """
    important_patterns = [
        r'name|title|label',
        r'date|time|start|end',
        r'competition|league|tournament',
        r'id|key|code',
        r'status|state|phase',
        r'score|result|outcome',
        r'type|category|group',
        r'location|venue|place'
    ]
    
    important_cols = []
    for col in df.columns:
        col_lower = col.lower()
        for pattern in important_patterns:
            if re.search(pattern, col_lower):
                important_cols.append(col)
                break
    
    return list(set(important_cols))  # Remove duplicates

def clean_numeric_string(value: str) -> float:
    """
    Clean and convert a string value to float.
    Handles various formats and removes any non-numeric characters.
    
    Args:
        value: String value to convert
        
    Returns:
        Float value or None if conversion fails
    """
    try:
        # Remove any non-numeric characters except decimal point
        cleaned = re.sub(r'[^\d.]', '', str(value))
        # Convert to float
        return float(cleaned)
    except (ValueError, TypeError):
        return None

def generate_query_explanation(sql_query: str) -> str:
    """
    Generate a human-readable explanation of the SQL query.
    
    Args:
        sql_query: The SQL query to explain
        
    Returns:
        A formatted explanation of the query
    """
    # Extract key components from the query
    table_match = re.search(r'FROM\s+(\w+)', sql_query, re.IGNORECASE)
    where_conditions = re.findall(r'WHERE\s+(.*?)(?:ORDER BY|LIMIT|$)', sql_query, re.IGNORECASE | re.DOTALL)
    order_by = re.search(r'ORDER BY\s+(.*?)(?:LIMIT|$)', sql_query, re.IGNORECASE | re.DOTALL)
    limit = re.search(r'LIMIT\s+(\d+)', sql_query, re.IGNORECASE)
    
    explanation = []
    
    # Add table information
    if table_match:
        explanation.append(f"This query retrieves data from the **{table_match.group(1)}** table.")
    
    # Add WHERE conditions
    if where_conditions:
        conditions = where_conditions[0].strip()
        explanation.append("It filters for the following conditions:")
        for condition in conditions.split('AND'):
            condition = condition.strip()
            if condition:
                explanation.append(f"- {condition}")
    
    # Add ORDER BY information
    if order_by:
        explanation.append(f"Results are ordered by: {order_by.group(1).strip()}")
    
    # Add LIMIT information
    if limit:
        explanation.append(f"Results are limited to {limit.group(1)} rows")
    
    return "\n".join(explanation)

def get_external_context(topic: str, subtopic: str = None, num_sentences: int = 6) -> str:
    """
    Get external context for a topic from Wikipedia and web sources.
    
    Args:
        topic: The main topic to search for
        subtopic: Optional subtopic for more specific information
        num_sentences: Number of sentences to include in the context (default: 6)
        
    Returns:
        A formatted string with external context
    """
    context = []
    
    try:
        import wikipedia
        from bs4 import BeautifulSoup
        from urllib.parse import quote
        import nltk
        from nltk.tokenize import sent_tokenize
        
        # Download NLTK data if needed
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        def get_wikipedia_summary(query: str, sentences: int = num_sentences) -> str:
            """Get a summary from Wikipedia for a given query with specific number of sentences."""
            try:
                # Search Wikipedia
                search_results = wikipedia.search(query, results=3)
                if search_results:
                    # Try each search result until we find a valid page
                    for result in search_results:
                        try:
                            # Get the page
                            page = wikipedia.page(result)
                            # Get the full summary
                            full_summary = page.summary
                            
                            # Extract specific number of sentences
                            all_sentences = sent_tokenize(full_summary)
                            limited_summary = ' '.join(all_sentences[:sentences])
                            
                            # Add source attribution
                            limited_summary += f" (Source: Wikipedia - {page.title})"
                            
                            return limited_summary
                        except wikipedia.exceptions.DisambiguationError:
                            continue
                        except wikipedia.exceptions.PageError:
                            continue
                return None
            except Exception as e:
                print(f"Wikipedia error: {str(e)}")
                return None
        
        def get_web_search(query: str) -> str:
            """Get relevant information from web search."""
            try:
                # Use DuckDuckGo API for web search
                url = f"https://api.duckduckgo.com/?q={quote(query)}&format=json"
                response = requests.get(url)
                if response.status_code == 200:
                    data = response.json()
                    if data.get('Abstract'):
                        # Extract specific number of sentences
                        all_sentences = sent_tokenize(data['Abstract'])
                        limited_summary = ' '.join(all_sentences[:num_sentences])
                        
                        # Add source attribution
                        if data.get('AbstractSource'):
                            limited_summary += f" (Source: {data['AbstractSource']})"
                        
                        return limited_summary
                return None
            except Exception as e:
                print(f"Web search error: {str(e)}")
                return None
        
        # Try to get context for the main topic
        if topic:
            # Try Wikipedia first
            wiki_summary = get_wikipedia_summary(topic)
            if wiki_summary:
                context.append(wiki_summary)
            else:
                # Fallback to web search
                web_info = get_web_search(topic)
                if web_info:
                    context.append(web_info)
        
        # Try to get additional context for the subtopic if provided
        if subtopic and subtopic != topic:
            # Try Wikipedia first for the combined query
            combined_query = f"{topic} {subtopic}"
            wiki_summary = get_wikipedia_summary(combined_query)
            if wiki_summary:
                context.append(wiki_summary)
            else:
                # Try just the subtopic
                wiki_summary = get_wikipedia_summary(subtopic)
                if wiki_summary:
                    context.append(wiki_summary)
                else:
                    # Fallback to web search
                    web_info = get_web_search(combined_query)
                    if web_info:
                        context.append(web_info)
        
        # If no context was found, add a note
        if not context:
            context.append("No additional context found from external sources.")
    
    except ImportError:
        # If required packages are not installed, show a message
        context.append("""
        Note: External context features require additional Python packages.
        Please install the following packages to enable this feature:
        ```
        pip install wikipedia-api beautifulsoup4 requests nltk
        ```
        """)
    
    return "\n".join(context)

def format_schema_for_prompt(schema: Optional[TableSchema]) -> str:
    """Formats the TableSchema object into a string suitable for AI prompts."""
    if not schema:
        return "Schema not available."

    schema_str = f"Table: {schema.name}\nColumns:\n"
    for col in schema.columns:
        col_info = f"  - {col['name']} ({col['type']})"
        if not col.get('nullable', True):
            col_info += " NOT NULL"
        if col.get('default') is not None:
            col_info += f" DEFAULT {col.get('default')}"
        schema_str += col_info + "\n"

    if schema.primary_keys:
        schema_str += f"Primary Keys: {', '.join(schema.primary_keys)}\n"

    if schema.foreign_keys:
        schema_str += "Foreign Keys:\n"
        for fk in schema.foreign_keys:
            schema_str += f"  - {', '.join(fk['constrained_columns'])} -> {fk['referred_table']}({', '.join(fk['referred_columns'])})\n"

    return schema_str

# Modified function signature: removed julius_client
def render_database_reasoning_ui(database_service: DatabaseService, ai_service: AIService, data_service: DataService, export_service: ExportService):
    """Render the database reasoning UI."""
    # Initialize session state variables if they don't exist
    if 'db_reasoning_chat_history' not in st.session_state:
        st.session_state.db_reasoning_chat_history = []
    if 'db_reasoning_mode' not in st.session_state:
        st.session_state.db_reasoning_mode = "simple"
    # Removed: Initialization of st.session_state.db_reasoning

    # Display connection status
    if not database_service.current_connection:
        st.error("Not connected to a database. Please connect to a database first.")
        return

    # Display the selected table
    if st.session_state.selected_table:
        st.info(f"Selected table: {st.session_state.selected_table}")
    else:
        st.error("No table selected. Please select a table from the sidebar.")
        return

    # Chat interface
    st.markdown("### Chat with Database")
    
    # Display chat history
    for message in st.session_state.db_reasoning_chat_history:
        if message["role"] == "user":
            # Display user messages normally
            with st.chat_message("user"):
                st.markdown(message["content"])
        else:
            # Display assistant messages using the custom display function
            with st.chat_message("assistant"):
                # Create a response object for the display function
                response = DatabaseReasoningResponse(
                    analysis=message["content"],
                    sql_query=message.get("sql_query", ""),
                    results=None,
                    code=message.get("code", None),
                    image_urls=message.get("image_urls", [])
                )
                
                # Get results if available
                results = None
                if message.get("results") is not None:
                    if isinstance(message["results"], pd.DataFrame) and not message["results"].empty:
                        results = message["results"]
                    elif isinstance(message["results"], str):
                        # Handle error messages
                        st.warning(f"Query Execution Note: {message['results']}")
                
                # Use the custom display function
                display_database_reasoning_response(response, results, st)
                
                # Add button to execute code if present
                if message.get("code"):
                    if st.button(f"Execute Code", key=f"execute_code_{message.get('timestamp', time.time())}"):
                        with st.spinner("Executing code..."):
                            try:
                                # Create outputs directory if it doesn't exist
                                os.makedirs("outputs", exist_ok=True)
                                
                                # Save the data to a temporary CSV file if results is a DataFrame
                                if isinstance(message.get("results"), pd.DataFrame):
                                    temp_csv = f"outputs/temp_data_{int(time.time())}.csv"
                                    message["results"].to_csv(temp_csv, index=False)
                                    
                                    # Modify the code to use the temporary CSV file
                                    modified_code = message["code"].replace(
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
                                    image_files = glob.glob("outputs/*.png") + glob.glob("outputs/*.jpg")
                                    
                                    if image_files:
                                        st.subheader("Generated Visualizations:")
                                        for img_file in image_files:
                                            # Only show images created in the last 10 seconds
                                            if os.path.getmtime(img_file) > time.time() - 10:
                                                st.image(img_file)
                            except Exception as e:
                                st.error(f"Error executing code: {str(e)}")
    
    # Chat input
    if prompt := st.chat_input("Ask a question about the database"):
        # Add user message to chat history
        st.session_state.db_reasoning_chat_history.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Process the query using AIService
        try:
            with st.spinner("Thinking and querying database..."):
                # Get schema and sample data for the selected table
                schema_obj = database_service.get_table_schema(st.session_state.selected_table)
                schema_str = format_schema_for_prompt(schema_obj)

                sample_data_str = "Sample data not available."
                try:
                    # Fetch sample data
                    success_sample, sample_df, error_sample = database_service.execute_query(
                        f"SELECT * FROM {st.session_state.selected_table} LIMIT 5"
                    )
                    if success_sample and sample_df is not None:
                         # Convert Polars DataFrame to string representation
                         sample_data_str = sample_df.to_pandas().to_string()
                    elif error_sample:
                         sample_data_str = f"Error fetching sample data: {error_sample}"
                except Exception as sample_e:
                    sample_data_str = f"Exception fetching sample data: {str(sample_e)}"


                # Determine if code generation is needed
                reasoning_mode = st.session_state.get("reasoning_mode", "simple")
                generate_code = (reasoning_mode == "advanced")

                # Create the request for AIService
                ai_request = DatabaseReasoningRequest(
                    query=prompt,
                    schema=schema_str,
                    table=st.session_state.selected_table,
                    sample_data=sample_data_str,
                    generate_code=generate_code
                )

                # Call AIService for reasoning and SQL generation
                ai_response = ai_service.database_reasoning(ai_request)

                # Execute the generated SQL query
                sql_to_execute = ai_response.sql_query
                results_df = None
                error_exec = None
                if sql_to_execute and "No SQL query generated" not in sql_to_execute:
                    success_exec, results_df, error_exec = database_service.execute_query(sql_to_execute)
                    if not success_exec:
                        st.error(f"Error executing generated SQL: {error_exec}")
                        # Store error message instead of DataFrame
                        results_df = f"Execution Error: {error_exec}"
                else:
                     results_df = "AI did not generate an SQL query."


                # Prepare assistant message content
                assistant_content = ai_response.analysis if ai_response.analysis else "Analysis complete."
                
                # Convert results to pandas DataFrame if it's a Polars DataFrame
                results_pandas = results_df.to_pandas() if isinstance(results_df, pl.DataFrame) else results_df
                
                # Create assistant message
                assistant_message = {
                    "role": "assistant",
                    "content": assistant_content,
                    "sql_query": ai_response.sql_query,
                    "results": results_pandas, # Store pandas df or error string
                    "code": ai_response.code,
                    "image_urls": ai_response.image_urls,
                    "timestamp": time.time()  # Add timestamp for unique button keys
                }

                # Add assistant response to chat history
                st.session_state.db_reasoning_chat_history.append(assistant_message)
                
                # Display the response using the custom display function
                with st.chat_message("assistant"):
                    display_database_reasoning_response(
                        ai_response,
                        results_pandas if isinstance(results_pandas, pd.DataFrame) else None,
                        st
                    )

                # Save any generated images to the outputs directory
                if ai_response.image_urls:
                    os.makedirs("outputs", exist_ok=True)
                    for i, url in enumerate(ai_response.image_urls):
                        if url.startswith("http"):
                            try:
                                import requests
                                from io import BytesIO
                                from PIL import Image as PILImage
                                
                                img_response = requests.get(url)
                                if img_response.status_code == 200:
                                    img = PILImage.open(BytesIO(img_response.content))
                                    save_path = os.path.join("outputs", f"db_reasoning_image_{int(time.time())}_{i}.png")
                                    img.save(save_path)
                                    # Add the local path to the image_urls list
                                    if save_path not in assistant_message["image_urls"]:
                                        assistant_message["image_urls"].append(save_path)
                            except Exception as e:
                                st.error(f"Error saving image: {str(e)}")

                # Rerun to display the new message immediately
                st.rerun()

        except Exception as e:
            st.error(f"Error processing query: {str(e)}")
            
            # Add error message to chat history
            st.session_state.db_reasoning_chat_history.append({
                "role": "assistant",
                "content": f"Error: {str(e)}"
            })