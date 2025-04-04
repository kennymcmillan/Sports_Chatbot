"""
Incremental Database Reasoning UI Module.
Provides real-time feedback during query processing.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from typing import Optional
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def render_incremental_db_reasoning(database_service, ai_service, data_service, export_service):
    """Render the incremental database reasoning UI."""
    
    st.header("Database Reasoning")
    
    # Initialize chat history if not exists
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display additional content for assistant messages
            if message["role"] == "assistant":
                if "sql" in message:
                    with st.expander("SQL Query"):
                        st.code(message["sql"], language="sql")
                if "results" in message:
                    st.dataframe(message["results"])
                if "viz" in message:
                    st.plotly_chart(message["viz"])
                if "code" in message:
                    with st.expander("Python Code"):
                        st.code(message["code"], language="python")
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your database..."):
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Create placeholders for assistant response
        with st.chat_message("assistant"):
            # Create placeholders for each component
            status_placeholder = st.empty()
            sql_placeholder = st.empty()
            results_placeholder = st.empty()
            viz_placeholder = st.empty()
            code_placeholder = st.empty()
            
            try:
                # 1. Initial status
                status_placeholder.info("Analyzing your question...")
                time.sleep(0.5)  # Small delay for UX
                
                # 2. Generate SQL
                status_placeholder.info("Generating SQL query...")
                sql_query = database_service.generate_sql(prompt)
                if sql_query:
                    sql_placeholder.code(sql_query, language="sql")
                    time.sleep(0.5)
                
                # 3. Execute query
                status_placeholder.info("Executing query...")
                results = database_service.execute_query(sql_query)
                if isinstance(results, pd.DataFrame):
                    results_placeholder.dataframe(results)
                    time.sleep(0.5)
                    
                    # 4. Generate visualizations if appropriate
                    if len(results) > 0:
                        status_placeholder.info("Generating visualizations...")
                        viz = database_service.generate_visualizations(results)
                        if viz:
                            viz_placeholder.plotly_chart(viz)
                        time.sleep(0.5)
                
                # 5. Generate code if in advanced mode
                if st.session_state.get("reasoning_mode") == "advanced":
                    status_placeholder.info("Generating analysis code...")
                    code = database_service.generate_code(prompt, sql_query, results)
                    if code:
                        code_placeholder.code(code, language="python")
                    time.sleep(0.5)
                
                # 6. Clear status and save to history
                status_placeholder.empty()
                
                # Add assistant response to history
                assistant_response = {
                    "role": "assistant",
                    "content": "Here are the results of your query:",
                    "sql": sql_query if sql_query else None,
                    "results": results if isinstance(results, pd.DataFrame) else None,
                    "viz": viz if 'viz' in locals() and viz else None,
                    "code": code if 'code' in locals() and code else None
                }
                st.session_state.chat_history.append(assistant_response)
                
            except Exception as e:
                logger.error(f"Error processing query: {str(e)}")
                status_placeholder.error(f"Error: {str(e)}")
                
                # Add error message to history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"I encountered an error: {str(e)}"
                })

def render_incremental_db_reasoning_sidebar():
    """Render the incremental database reasoning sidebar."""
    
    st.sidebar.header("Database Settings")
    
    # Reasoning mode selection
    st.sidebar.subheader("Reasoning Mode")
    mode = st.sidebar.radio(
        "Select mode:",
        ["Simple", "Advanced"],
        help="Simple mode focuses on queries and results. Advanced mode includes visualizations and code generation."
    )
    st.session_state.reasoning_mode = mode.lower()
    
    # Table selection if connected
    if st.session_state.get('db_connected'):
        st.sidebar.subheader("Table Selection")
        tables = database_service.get_tables()
        selected_table = st.sidebar.selectbox(
            "Focus on table:",
            ["All Tables"] + tables,
            help="Select a specific table to focus your queries on"
        )
        if selected_table != "All Tables":
            st.session_state.selected_table = selected_table
        else:
            st.session_state.selected_table = None 