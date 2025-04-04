"""
Query Builder UI Module for Multi-Source Julius AI Chatbot.

This module provides UI components for building SQL queries using natural language and direct SQL input.
"""

import os
import streamlit as st
import pandas as pd
import polars as pl
from typing import Optional, Dict, List, Any, Tuple

from core_services.database_service import DatabaseService
from core_services.data_service import DataService
from core_services.export_service import ExportService
# Removed: from ui_components.julius_adapter import generate_sql_with_julius, SQLGenerationResponse
# Assuming ai_service is passed to render_nl_sql_tab now
from core_services.ai_service import AIService, SQLGenerationRequest # Added AIService import

# Try to import streamlit-ace for SQL editor
try:
    from streamlit_ace import st_ace
    ACE_AVAILABLE = True
except ImportError:
    ACE_AVAILABLE = False

def render_query_builder_sidebar(database_service: DatabaseService):
    """
    Render the query builder sidebar, showing schema for the selected table.

    Args:
        database_service: DatabaseService instance
    """
    st.markdown("### Query Builder")

    # Check if connected and a table is selected
    if not st.session_state.get('db_connected', False) or not st.session_state.get('selected_table'):
        st.info("Connect to a database and select a table in the 'Data Source' section first.")
        return

    selected_table = st.session_state.get('selected_table')
    current_connection = database_service.current_connection # Assumes this is set correctly in app.py

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
            if st.button(f"Show Sample Data for {selected_table}", key="qb_show_sample"):
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


# Modified function signature: replaced julius with ai_service
def render_query_builder(database_service: DatabaseService, ai_service: AIService,
                         data_service: DataService, export_service: ExportService):
    """
    Render the query builder main content.

    Args:
        database_service: DatabaseService instance
        ai_service: AIService instance
        data_service: DataService instance
        export_service: ExportService instance
    """
    st.header("Query Builder")

    # Check if connected to database and table selected
    if not st.session_state.get('db_connected', False):
        st.warning("Please connect to a database using the 'Data Source' section in the sidebar.")
        return
    if not st.session_state.get('selected_table'):
         st.warning("Please select a table using the 'Data Source' section in the sidebar.")
         return

    # Create tabs for different query methods
    tab1, tab2 = st.tabs(["Direct SQL", "Natural Language"])

    with tab1:
        render_direct_sql_tab(database_service, data_service, export_service)

    with tab2:
        # Pass ai_service instead of julius
        render_nl_sql_tab(database_service, ai_service, data_service, export_service)

def render_direct_sql_tab(database_service: DatabaseService, data_service: DataService,
                          export_service: ExportService):
    """
    Render the direct SQL tab.

    Args:
        database_service: DatabaseService instance
        data_service: DataService instance
        export_service: ExportService instance
    """
    st.markdown("### Direct SQL")
    selected_table = st.session_state.get('selected_table', 'your_table') # Default for placeholder

    # SQL editor
    if ACE_AVAILABLE:
        # Use streamlit-ace for SQL editor with syntax highlighting
        sql_query = st_ace(
            value=f"SELECT * FROM {selected_table} LIMIT 10;",
            language="sql",
            theme="monokai",
            auto_update=True,
            key="sql_editor"
        )
    else:
        # Fallback to regular text area
        sql_query = st.text_area(
            "SQL Query",
            value=f"SELECT * FROM {selected_table} LIMIT 10;",
            height=200,
            key="sql_query"
        )

    # Execute button
    if st.button("Execute Query", key="execute_direct_sql"):
        if sql_query:
            with st.spinner("Executing query..."):
                # Execute query
                success, result, error = database_service.execute_query(sql_query)

                if success:
                    if result is not None and not result.is_empty(): # Check if polars df is not empty
                        # Store result in session state
                        st.session_state['query_result'] = result
                        st.session_state['query_sql'] = sql_query

                        # Display result summary
                        st.success(f"Query executed successfully. {result.height} rows returned.")

                        # Display result
                        st.markdown("#### Query Result")
                        st.dataframe(result.to_pandas()) # Display as pandas for better streamlit rendering

                        # Save as dataset option
                        st.markdown("#### Save as Dataset")

                        col1, col2 = st.columns([3, 1])
                        with col1:
                            dataset_name = st.text_input("Dataset Name", f"{selected_table}_query_result.csv", key="direct_dataset_name")
                        with col2:
                            # Add margin to button
                            st.markdown('<style>div.stButton > button {margin-top: 28px;}</style>', unsafe_allow_html=True)
                            if st.button("Save as Dataset", key="direct_save_dataset"):
                                # Save as dataset using DataService (expects polars df)
                                success_save, _, error_save = data_service.save_dataframe(result, dataset_name)

                                if success_save:
                                    st.success(f"Query result saved as dataset: {dataset_name}")
                                else:
                                    st.error(f"Failed to save dataset: {error_save}")

                        # Export options
                        st.markdown("#### Export Options")

                        col1, col2, col3 = st.columns(3)

                        with col1:
                            # CSV export
                            try:
                                csv_data = export_service.export_csv(result)
                                st.download_button(
                                    label="Download CSV",
                                    data=csv_data,
                                    file_name=f"{selected_table}_query_result.csv",
                                    mime="text/csv",
                                    key="direct_download_csv"
                                )
                            except Exception as e:
                                st.error(f"CSV Export Error: {e}")

                        with col2:
                             # Excel export
                            try:
                                excel_data = export_service.export_excel(result)
                                st.download_button(
                                    label="Download Excel",
                                    data=excel_data,
                                    file_name=f"{selected_table}_query_result.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    key="direct_download_excel"
                                )
                            except Exception as e:
                                st.error(f"Excel Export Error: {e}")

                        with col3:
                            # JSON export
                            try:
                                json_data = export_service.export_json(result)
                                st.download_button(
                                    label="Download JSON",
                                    data=json_data,
                                    file_name=f"{selected_table}_query_result.json",
                                    mime="application/json",
                                    key="direct_download_json"
                                )
                            except Exception as e:
                                st.error(f"JSON Export Error: {e}")
                    else:
                        st.info("Query executed successfully. No results returned.")
                else:
                    st.error(f"Error executing query: {error}")
        else:
            st.warning("Please enter a SQL query.")

# Modified function signature: ai_service type hint added
def render_nl_sql_tab(database_service: DatabaseService, ai_service: AIService,
                      data_service: DataService, export_service: ExportService):
    """
    Render the natural language to SQL tab.

    Args:
        database_service: DatabaseService instance
        ai_service: AIService instance
        data_service: DataService instance
        export_service: ExportService instance
    """
    st.markdown("### Natural Language to SQL")

    # Check if connected to database and table selected (already done in parent function)
    current_table = st.session_state.get('selected_table')
    if not current_table:
         # This check might be redundant if parent function handles it, but good for safety
         st.warning("Please select a table using the 'Data Source' section in the sidebar.")
         return

    # Get table schema
    schema_obj = database_service.get_table_schema(current_table)
    if not schema_obj:
        st.error(f"Could not retrieve schema for table: {current_table}")
        return
    # Format schema for prompt (assuming a helper function exists or we create one)
    from ui_components.database_reasoning_ui import format_schema_for_prompt # Reuse from other UI module
    schema_str = format_schema_for_prompt(schema_obj)


    # Get sample data
    sample_data_str = "Sample data not available."
    try:
        success_sample, sample_df, error_sample = database_service.execute_query(
            f"SELECT * FROM {current_table} LIMIT 5"
        )
        if success_sample and sample_df is not None:
            sample_data_str = sample_df.to_pandas().to_string()
        elif error_sample:
            sample_data_str = f"Error fetching sample data: {error_sample}"
    except Exception as sample_e:
        sample_data_str = f"Exception fetching sample data: {str(sample_e)}"


    # Get user input
    query = st.text_area("Enter your query in natural language", key="nl_query_input")

    # Store generated SQL in session state to persist across reruns after button clicks
    if 'generated_sql' not in st.session_state:
        st.session_state.generated_sql = None
    if 'sql_interpretation' not in st.session_state:
        st.session_state.sql_interpretation = None
    if 'sql_explanation' not in st.session_state:
        st.session_state.sql_explanation = None


    if st.button("Generate SQL", key="generate_sql_btn"):
        if not query:
            st.warning("Please enter a query.")
            # Clear previous results if query is empty
            st.session_state.generated_sql = None
            st.session_state.sql_interpretation = None
            st.session_state.sql_explanation = None
            st.session_state.nl_query_result = None # Clear results too
            return
        else:
            # Clear previous results before generating new ones
            st.session_state.generated_sql = None
            st.session_state.sql_interpretation = None
            st.session_state.sql_explanation = None
            st.session_state.nl_query_result = None # Clear results too

            with st.spinner("Generating SQL..."):
                try:
                    # Get current connection's dialect
                    if not database_service.current_connection:
                        st.error("No active database connection.")
                        return

                    connection_info = database_service.connections.get(database_service.current_connection)
                    if not connection_info:
                        st.error("Could not retrieve connection information.")
                        return

                    dialect = connection_info.get('type', 'mysql')  # Default to mysql

                    # Create request
                    request = SQLGenerationRequest(
                        query=query,
                        schema=schema_str, # Use formatted schema string
                        dialect=dialect,
                        table=current_table,
                        sample_data=sample_data_str
                    )

                    # Generate SQL using AIService
                    response = ai_service.generate_sql(request)

                    # Store results in session state
                    st.session_state.generated_sql = response.sql
                    st.session_state.sql_interpretation = response.interpretation
                    st.session_state.sql_explanation = response.explanation

                except Exception as e:
                    st.error(f"Error generating SQL: {str(e)}")
                    # Clear potentially partially stored results on error
                    st.session_state.generated_sql = None
                    st.session_state.sql_interpretation = None
                    st.session_state.sql_explanation = None


    # Display generated SQL if available in session state
    if st.session_state.generated_sql:
        st.markdown("---")
        st.markdown("### Generated SQL")
        st.code(st.session_state.generated_sql, language="sql")

        if st.session_state.sql_interpretation:
            st.markdown("### Interpretation")
            st.write(st.session_state.sql_interpretation)

        if st.session_state.sql_explanation:
            st.markdown("### Explanation")
            st.write(st.session_state.sql_explanation)

        # Add execute button
        if st.button("Execute Generated SQL", key="execute_nl_sql"):
             with st.spinner("Executing generated query..."):
                success, result, error = database_service.execute_query(st.session_state.generated_sql)

                if success:
                    if result is not None and not result.is_empty():
                         st.session_state.nl_query_result = result # Store result
                         st.success(f"Query executed successfully. {result.height} rows returned.")
                    else:
                         st.session_state.nl_query_result = None # Clear result if empty
                         st.info("Query executed successfully. No results returned.")
                else:
                    st.session_state.nl_query_result = None # Clear result on error
                    st.error(f"Error executing query: {error}")

    # Display results if available in session state
    if 'nl_query_result' in st.session_state and st.session_state.nl_query_result is not None:
         st.markdown("---")
         st.markdown("### Query Results")
         st.dataframe(st.session_state.nl_query_result.to_pandas()) # Display as pandas

         # Export options for NL query results
         st.markdown("#### Export Options")
         col1, col2, col3 = st.columns(3)
         result_df_nl = st.session_state.nl_query_result # Use stored result

         with col1:
             try:
                 csv_data_nl = export_service.export_csv(result_df_nl)
                 st.download_button(
                     label="Download CSV",
                     data=csv_data_nl,
                     file_name=f"{current_table}_nl_query_result.csv",
                     mime="text/csv",
                     key="nl_download_csv"
                 )
             except Exception as e:
                 st.error(f"CSV Export Error: {e}")
         with col2:
             try:
                 excel_data_nl = export_service.export_excel(result_df_nl)
                 st.download_button(
                     label="Download Excel",
                     data=excel_data_nl,
                     file_name=f"{current_table}_nl_query_result.xlsx",
                     mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                     key="nl_download_excel"
                 )
             except Exception as e:
                 st.error(f"Excel Export Error: {e}")
         with col3:
             try:
                 json_data_nl = export_service.export_json(result_df_nl)
                 st.download_button(
                     label="Download JSON",
                     data=json_data_nl,
                     file_name=f"{current_table}_nl_query_result.json",
                     mime="application/json",
                     key="nl_download_json"
                 )
             except Exception as e:
                 st.error(f"JSON Export Error: {e}")