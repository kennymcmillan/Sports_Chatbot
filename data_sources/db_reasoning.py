"""
Database Reasoning Module for Multi-Source Julius AI Chatbot.

This module provides functionality to reason directly over database connections
using Julius AI, without requiring users to first load data as a dataset.
"""

import os
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import re
import streamlit as st
import time
import os
from data_sources.sql_connector import SQLConnector
from julius_api import Julius
from core_services.database_service import DatabaseService
from typing import Union

class DatabaseReasoning:
    """
    A class to handle direct database reasoning using Julius AI.
    
    This class provides methods to:
    - Get database schema information
    - Send queries to Julius AI with database context
    - Execute generated SQL queries
    - Format results for display
    """
    
    def __init__(self, sql_connector: Union[SQLConnector, Any] = None, database_service: DatabaseService = None, julius: Julius = None):
        """
        Initialize the Database Reasoning module.
        
        Args:
            sql_connector: SQLConnector instance or compatible adapter
            database_service: DatabaseService instance (deprecated, use sql_connector instead)
            julius: Julius API client instance
        """
        # Support both direct SQLConnector and adapter pattern
        self.sql_connector = sql_connector
        self.julius = julius
        
        # For backward compatibility
        if database_service is not None and sql_connector is None:
            import warnings
            warnings.warn("Using database_service directly is deprecated. Use sql_connector instead.", DeprecationWarning)
            from data_sources.sql_connector_adapter import SQLConnectorAdapter
            self.sql_connector = SQLConnectorAdapter(database_service)
        
        self._verify_connection()
        
    def _verify_connection(self) -> bool:
        """
        Verify that the database connection is active.
        
        Returns:
            bool: True if connection is active, False otherwise
        """
        if not hasattr(self, 'sql_connector') or self.sql_connector is None:
            return False
            
        return getattr(self.sql_connector, 'connected', False)
            
    def ensure_connection(self) -> Tuple[bool, str]:
        """
        Ensure database connection is active, attempt to reconnect if needed.
        
        Returns:
            Tuple[bool, str]: (success, error_message)
        """
        if self._verify_connection():
            return True, ""
            
        # Try to reconnect
        st.info("Attempting to connect to database...")
        start_time = time.time()
        
        try:
            success = self.sql_connector.connect_to_database(use_env=True)
            connection_time = time.time() - start_time
            
            if success:
                st.success(f"✅ Connected to database successfully in {connection_time:.2f} seconds")
                return True, ""
            else:
                st.error(f"❌ Failed to connect to database")
                return False, "Failed to connect to database"
        except Exception as e:
            st.error(f"❌ Failed to connect to database: {str(e)}")
            return False, f"Failed to connect to database: {str(e)}"
            return False, f"Failed to connect to database: {str(e)}"
        
    def get_database_schema(self) -> str:
        """
        Get the database schema information.
        
        Returns:
            str: Formatted database schema information
        """
        success, error = self.ensure_connection()
        if not success:
            return f"Database connection error: {error}"
            
        try:
            st.info("Fetching database schema...")
            start_time = time.time()
            
            # Get list of tables
            tables = self.sql_connector.get_tables()
            
            if not tables:
                st.warning("No tables found in the database")
                return "No tables found in the database"
            
            # Build schema information
            schema_info = "Database Schema:\n\n"
            
            for table in tables:
                # Get table schema
                schema_str = self.sql_connector.get_table_info(table)
                if schema_str:
                    schema_info += schema_str + "\n\n"
                
                # Get sample data (first 5 rows)
                try:
                    sample_query = f"SELECT * FROM {table} LIMIT 5"
                    success, result_df, error = self.sql_connector.execute_direct_sql(sample_query)
                    if success and result_df is not None and not result_df.empty:
                        schema_info += f"Sample data:\n{result_df.to_string()}\n\n"
                except Exception as e:
                    st.warning(f"Error getting sample data for table {table}: {str(e)}")
            
            schema_time = time.time() - start_time
            st.success(f"✅ Schema fetched successfully in {schema_time:.2f} seconds")
            return schema_info
        except Exception as e:
            st.error(f"Error getting database schema: {str(e)}")
            return f"Error getting database schema: {str(e)}"
    
    def query_database(self, user_query: str, selected_table: str = None, generate_code: bool = False) -> Tuple[bool, Dict[str, Any]]:
        """
        Query the database using natural language.
        
        Args:
            user_query: Natural language query
            selected_table: Optional specific table to query
            generate_code: Whether to generate Python code
            
        Returns:
            Tuple containing:
            - bool: Success status
            - Dict[str, Any]: Response data or error message
        """
        try:
            # Ensure we have a database connection
            success, error = self.ensure_connection()
            if not success:
                return False, {"error": error}
            
            # Get database schema
            schema = self.get_database_schema()
            if not schema:
                return False, {"error": "Failed to get database schema"}
            
            # Build prompt for Julius
            if selected_table:
                prompt = f"""
                Given the following database schema for table '{selected_table}':

                {schema}

                Please generate a SQL query to answer this question: {user_query}
                
                The query should:
                1. Be valid MySQL syntax
                2. Use proper table and column names
                3. Include appropriate WHERE clauses
                4. Use proper LIKE syntax with single % for wildcards
                5. Include ORDER BY and LIMIT clauses when appropriate
                6. Keep the entire query on a single line
                7. Do NOT include line breaks in the query
                8. Do NOT include extra commas between column names
                9. Format column lists like: col1, col2, col3
                10. Do NOT include trailing commas in column lists
                
                Return only the SQL query, nothing else.
                """
            else:
                prompt = f"""
                Given the following database schema:

                {schema}

                Please generate a SQL query to answer this question: {user_query}
                
                The query should:
                1. Be valid MySQL syntax
                2. Use proper table and column names
                3. Include appropriate WHERE clauses
                4. Use proper LIKE syntax with single % for wildcards
                5. Include ORDER BY and LIMIT clauses when appropriate
                6. Keep the entire query on a single line
                7. Do NOT include line breaks in the query
                8. Do NOT include extra commas between column names
                9. Format column lists like: col1, col2, col3
                10. Do NOT include trailing commas in column lists
                
                Return only the SQL query, nothing else.
                """
            
            # Get SQL query from Julius
            sql_query = self.julius.generate_sql(prompt)
            if not sql_query:
                return False, {"error": "Failed to generate SQL query"}
            
            # Clean up the SQL query
            sql_query = sql_query.strip()
            if sql_query.startswith("```sql"):
                sql_query = sql_query[6:]
            if sql_query.endswith("```"):
                sql_query = sql_query[:-3]
            sql_query = sql_query.strip()
            
            # Fix LIKE clause syntax if present
            sql_query = sql_query.replace("%%", "%")
            
            # Format the query to keep everything on a single line and fix any syntax issues
            # Remove all line breaks and extra whitespace
            sql_query = ' '.join(sql_query.split())
            
            # Fix common syntax issues with MySQL
            # Replace multiple commas with single commas
            sql_query = re.sub(r',\s+,', ',', sql_query)
            
            # Remove any trailing commas in column lists
            sql_query = re.sub(r',\s*FROM', ' FROM', sql_query, flags=re.IGNORECASE)
            
            # Fix column list formatting - this is the most critical part
            # First, identify the SELECT and FROM parts
            select_match = re.match(r'(SELECT\s+)(.*?)(\s+FROM\s+.*)', sql_query, re.IGNORECASE)
            if select_match:
                select_keyword = select_match.group(1)  # SELECT
                columns_part = select_match.group(2)    # column1, column2, etc.
                from_part = select_match.group(3)       # FROM table WHERE...
                
                # Clean up the columns part
                columns = [col.strip() for col in columns_part.split(',')]
                clean_columns = ', '.join(columns)
                
                # Reconstruct the query
                sql_query = f"{select_keyword}{clean_columns}{from_part}"
            
            # Execute query using sql_connector
            success, result_df, error = self.sql_connector.execute_direct_sql(sql_query)
            if not success:
                return False, {"error": f"Failed to execute SQL query: {error}"}
            
            # Build response
            response = {
                "response": "Here are the results of your query:",
                "sql_query": sql_query,
                "results": result_df
            }
            
            # Generate code if requested
            if generate_code and result_df is not None:
                code_prompt = f"""
                Given the following SQL query and its results:

                SQL Query:
                {sql_query}

                Please generate Python code to:
                1. Execute this query
                2. Process the results
                3. Create visualizations if appropriate
                
                Return only the Python code, nothing else.
                """
                
                code = self.julius.generate_code(code_prompt)
                if code:
                    response["code"] = code
            
            return True, response
            
        except Exception as e:
            return False, {"error": str(e)}
    
    def _extract_sql_query(self, text: str) -> Optional[str]:
        """
        Extract SQL query from text.
        
        Args:
            text: Text containing SQL query
            
        Returns:
            Optional[str]: Extracted SQL query or None
        """
        # Look for SQL code blocks
        
        # Try to find SQL in markdown code blocks
        sql_pattern = r'```sql\s*(.*?)\s*```'
        matches = re.findall(sql_pattern, text, re.DOTALL)
        if matches:
            # Clean up the query to ensure it's valid
            query = matches[0].strip()
            # Remove any trailing 'the' that might be accidentally added
            if query.lower().endswith(' the'):
                query = query[:-4]
            return query
        
        # Try to find generic code blocks that might contain SQL
        code_pattern = r'```\s*(.*?)\s*```'
        matches = re.findall(code_pattern, text, re.DOTALL)
        for match in matches:
            if 'SELECT' in match.upper() or 'FROM' in match.upper():
                # Clean up the query to ensure it's valid
                query = match.strip()
                # Remove any trailing 'the' that might be accidentally added
                if query.lower().endswith(' the'):
                    query = query[:-4]
                return query
        
        # Try to find SQL queries without code blocks
        if 'SELECT' in text.upper() and 'FROM' in text.upper():
            lines = text.split('\n')
            for i, line in enumerate(lines):
                if 'SELECT' in line.upper():
                    # Extract the query starting from this line
                    query_lines = []
                    j = i
                    while j < len(lines) and ';' not in lines[j]:
                        query_lines.append(lines[j])
                        j += 1
                    if j < len(lines):
                        query_lines.append(lines[j])
                    
                    # Clean up the query to ensure it's valid
                    query = '\n'.join(query_lines).strip()
                    # Remove any trailing 'the' that might be accidentally added
                    if query.lower().endswith(' the'):
                        query = query[:-4]
                    return query
        
        return None
        
    def _extract_image_urls(self, text: str) -> List[str]:
        """
        Extract image URLs from text.
        
        Args:
            text: Text containing image URLs
            
        Returns:
            List[str]: List of extracted image URLs
        """
        image_urls = []
        
        # Look for image URLs in markdown format
        markdown_pattern = r'!\[.*?\]\((https?://[^\s)]+)\)'
        markdown_matches = re.findall(markdown_pattern, text)
        image_urls.extend(markdown_matches)
        
        # Look for image URLs in HTML format
        html_pattern = r'<img[^>]*src=[\'"]([^\'"]+)[\'"][^>]*>'
        html_matches = re.findall(html_pattern, text)
        image_urls.extend(html_matches)
        
        # Look for direct URLs that end with image extensions
        url_pattern = r'(https?://[^\s]+\.(?:png|jpg|jpeg|gif|bmp))'
        url_matches = re.findall(url_pattern, text)
        image_urls.extend(url_matches)
        
        # Look for image URLs in JSON format
        json_pattern = r'"image_urls":\s*\[(.*?)\]'
        json_matches = re.findall(json_pattern, text, re.DOTALL)
        if json_matches:
            for match in json_matches:
                url_pattern = r'"(https?://[^"]+)"'
                urls = re.findall(url_pattern, match)
                image_urls.extend(urls)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_urls = []
        for url in image_urls:
            if url not in seen:
                seen.add(url)
                unique_urls.append(url)
        
        return unique_urls
        
    def _extract_python_code(self, text: str) -> Optional[str]:
        """
        Extract Python code from text.
        
        Args:
            text: Text containing Python code
            
        Returns:
            Optional[str]: Extracted Python code or None
        """
        # Try to find Python in markdown code blocks
        python_pattern = r'```python\s*(.*?)\s*```'
        matches = re.findall(python_pattern, text, re.DOTALL)
        if matches:
            return matches[0].strip()
        
        # Try to find generic code blocks that might contain Python
        code_pattern = r'```\s*(.*?)\s*```'
        matches = re.findall(code_pattern, text, re.DOTALL)
        for match in matches:
            # Check if it looks like Python code
            if 'import' in match or 'def ' in match or 'class ' in match or '=' in match:
                return match.strip()
        
        return None