"""
SQL Connector Module for Julius AI Chatbot.

This module provides functionality to connect to SQL databases,
execute queries, and retrieve data using LangChain's SQL agent.
"""
import os
import pandas as pd
import ssl
from typing import Dict, List, Optional, Any, Tuple
import streamlit as st

# LangChain imports
try:
    from langchain.agents import create_sql_agent
    from langchain.agents.agent_toolkits import SQLDatabaseToolkit
    from langchain.sql_database import SQLDatabase
    from langchain_openai import ChatOpenAI
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    

class SQLConnector:
    """
    A class to handle SQL database connections and queries.
    
    This class provides methods to:
    - Connect to various SQL databases (MySQL, PostgreSQL, etc.)
    - Execute natural language queries using LangChain's SQL agent
    - Convert query results to pandas DataFrames
    """
    # Database connection constants
    DB_HOST = "sportsdb-sports-database-for-web-scrapes.g.aivencloud.com"
    DB_PORT = 16439
    DB_USER = "avnadmin"
    DB_NAME = "defaultdb"
    SSL_ARGS = {'cert_reqs': ssl.CERT_NONE}  # Disable SSL verification (for testing only)
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the SQL Connector.
        
        Args:
            api_key: OpenAI API key for LangChain's SQL agent
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.db_connection = None
        self.agent = None
        self.connected = False
        self.connection_info = {}
        self.db_password = os.getenv("DB_PASSWORD")
        self.connection_info = {}
        
        # Check if LangChain is available
        if not LANGCHAIN_AVAILABLE:
            st.warning("LangChain is not installed. SQL functionality will be limited.")
    
    def connect_to_database(self,
                           db_type: str = None,
                           host: str = None,
                           user: str = None,
                           password: str = None,
                           database: str = None,
                           port: Optional[int] = None,
                           use_env: bool = False) -> bool:
        """
        Connect to a SQL database.
        
        Args:
            db_type: Type of database ('mysql', 'postgresql', etc.)
            host: Database host
            user: Database user
            password: Database password
            database: Database name
            port: Database port (optional)
            use_env: Whether to use environment variables for connection details
            
        Returns:
            bool: True if connection successful, False otherwise
        """
        if not LANGCHAIN_AVAILABLE:
            st.error("LangChain is not installed. Cannot connect to database.")
            return False
            
        # Use environment variables if specified
        if use_env:
            db_type = "mysql"  # Default to MySQL
            host = self.DB_HOST
            port = self.DB_PORT
            database = self.DB_NAME
            user = self.DB_USER
            password = self.db_password
            
            if not password:
                st.error("Database password not found in environment variables.")
                return False
        
        try:
            # Construct database URI based on database type
            if db_type.lower() == 'mysql':
                port = port or 3306
                db_uri = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
            elif db_type.lower() == 'postgresql':
                port = port or 5432
                db_uri = f"postgresql://{user}:{password}@{host}:{port}/{database}"
            elif db_type.lower() == 'sqlite':
                db_uri = f"sqlite:///{database}"
            else:
                st.error(f"Unsupported database type: {db_type}")
                return False
                
            # Create database connection
            # For MySQL with SSL, we need to handle the connection differently
            if db_type.lower() == 'mysql':
                try:
                    # Import pymysql for direct connection
                    import pymysql
                    
                    # Create a direct connection to test connectivity
                    conn = pymysql.connect(
                        host=host,
                        port=port,
                        user=user,
                        password=password,
                        database=database,
                        ssl_disabled=True,  # Disable SSL for testing
                        connect_timeout=10,
                        read_timeout=30,
                        write_timeout=30
                    )
                    
                    # Close the test connection
                    conn.close()
                    
                    # If connection test succeeds, create the SQLDatabase
                    # Add SSL parameters to the URI
                    db_uri = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}?ssl_disabled=true&connect_timeout=30&read_timeout=30"
                    
                    # Create the SQLDatabase with engine_kwargs
                    from sqlalchemy import create_engine
                    self.engine = create_engine(db_uri)
                    self.db_connection = SQLDatabase(engine=self.engine)
                    
                    # Store the direct connection for fallback
                    self.direct_connection = conn
                    
                    # Print success message with connection details
                    print(f"Successfully connected to MySQL database: {host}:{port}/{database}")
                    
                except Exception as e:
                    # Print detailed error for debugging
                    import traceback
                    print(f"MySQL connection error: {str(e)}")
                    traceback.print_exc()
                    raise Exception(f"MySQL connection error: {str(e)}")
            else:
                # For other database types
                self.db_connection = SQLDatabase.from_uri(db_uri)
            
            # Store connection info (excluding password)
            self.connection_info = {
                'db_type': db_type,
                'host': host,
                'user': user,
                'database': database,
                'port': port
            }
            
            # Initialize LLM for SQL agent
            llm = ChatOpenAI(
                temperature=0,
                api_key=self.api_key,
                model="gpt-4o"
            )
            
            # Create SQL agent
            toolkit = SQLDatabaseToolkit(db=self.db_connection, llm=llm)
            self.agent = create_sql_agent(
                llm=llm,
                toolkit=toolkit,
                verbose=True,
                agent_type="openai-tools"
            )
            
            self.connected = True
            return True
            
        except Exception as e:
            # Show detailed error message
            import traceback
            error_msg = f"Error connecting to database: {str(e)}"
            st.error(error_msg)
            st.code(traceback.format_exc())
            print(error_msg)
            traceback.print_exc()
            self.connected = False
            return False
    
    def execute_query(self, query: str) -> Tuple[bool, Optional[pd.DataFrame], Optional[str]]:
        """
        Execute a natural language query using LangChain's SQL agent.
        
        Args:
            query: Natural language query
            
        Returns:
            Tuple containing:
            - bool: Success status
            - Optional[pd.DataFrame]: Query results as DataFrame (if successful)
            - Optional[str]: Error message (if unsuccessful)
        """
        if not self.connected:
            return False, None, "Not connected to a database"
            
        try:
            # If agent is available, use LangChain's SQL agent
            if self.agent:
                try:
                    # Execute query using LangChain's SQL agent
                    result = self.agent.invoke({"input": query})
                    
                    # Extract SQL query from agent's response
                    sql_query = self._extract_sql_query(result)
                    
                    # Execute the SQL query directly to get a DataFrame
                    if sql_query:
                        print(f"Generated SQL query: {sql_query}")
                        df = self._execute_raw_sql(sql_query)
                        return True, df, None
                    else:
                        # If we couldn't extract the SQL query, try to parse the agent's response
                        return self._parse_agent_response(result)
                except Exception as agent_error:
                    # If LangChain agent fails, try to generate a simple SQL query
                    print(f"LangChain agent error: {str(agent_error)}")
                    
                    # Try to generate a simple SQL query based on the natural language query
                    if "all rows" in query.lower() and "from" in query.lower():
                        # Extract table name
                        table_parts = query.lower().split("from")
                        if len(table_parts) > 1:
                            table_name = table_parts[1].strip().split()[0].strip()
                            sql_query = f"SELECT * FROM {table_name} LIMIT 1000"
                            print(f"Generated simple SQL query: {sql_query}")
                            return self.execute_direct_sql(sql_query)
                    elif "first 10" in query.lower() and "from" in query.lower():
                        # Extract table name
                        table_parts = query.lower().split("from")
                        if len(table_parts) > 1:
                            table_name = table_parts[1].strip().split()[0].strip()
                            sql_query = f"SELECT * FROM {table_name} LIMIT 10"
                            print(f"Generated simple SQL query: {sql_query}")
                            return self.execute_direct_sql(sql_query)
                    elif "count" in query.lower() and "rows" in query.lower() and "from" in query.lower():
                        # Extract table name
                        table_parts = query.lower().split("from")
                        if len(table_parts) > 1:
                            table_name = table_parts[1].strip().split()[0].strip()
                            sql_query = f"SELECT COUNT(*) FROM {table_name}"
                            print(f"Generated simple SQL query: {sql_query}")
                            return self.execute_direct_sql(sql_query)
            
            # If we get here, we couldn't execute the query
            return False, None, "Could not parse natural language query. Try using direct SQL syntax."
                
        except Exception as e:
            # Show detailed error message
            import traceback
            error_msg = f"Error executing query: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            return False, None, error_msg
            
    def execute_direct_sql(self, sql_query: str) -> Tuple[bool, Optional[pd.DataFrame], Optional[str]]:
        """
        Execute a direct SQL query.
        
        Args:
            sql_query: SQL query to execute
            
        Returns:
            Tuple containing:
            - bool: Success status
            - Optional[pd.DataFrame]: Query results as DataFrame (if successful)
            - Optional[str]: Error message (if unsuccessful)
        """
        # Clean up the SQL query to ensure it's valid
        # Remove line breaks and extra whitespace
        sql_query = ' '.join(sql_query.split())
        
        # Fix common syntax issues with MySQL
        # Replace multiple commas with single commas
        import re
        sql_query = re.sub(r',\s+,', ',', sql_query)
        
        # Remove any trailing commas in column lists
        sql_query = re.sub(r',\s*FROM', ' FROM', sql_query, flags=re.IGNORECASE)
        
        try:
            # Try using SQLAlchemy engine directly
            import pandas as pd
            df = pd.read_sql_query(sql_query, self.engine)
            return True, df, None
        except Exception as e:
            # If SQLAlchemy fails, try using direct pymysql connection
            try:
                if hasattr(self, 'direct_connection') and self.direct_connection:
                    # Reconnect if needed
                    if not self.direct_connection.open:
                        self.direct_connection.connect()
                    
                    # Clean up the SQL query again to be sure
                    sql_query = ' '.join(sql_query.split())
                    
                    # Execute query
                    cursor = self.direct_connection.cursor()
                    cursor.execute(sql_query)
                    
                    # Fetch results
                    results = cursor.fetchall()
                    
                    # Get column names
                    columns = [column[0] for column in cursor.description]
                    
                    # Create DataFrame
                    df = pd.DataFrame(results, columns=columns)
                    
                    # Close cursor
                    cursor.close()
                    
                    return True, df, None
            except Exception as direct_error:
                # Show detailed error message
                import traceback
                error_msg = f"Error executing direct SQL query: {str(direct_error)}"
                print(error_msg)
                traceback.print_exc()
                return False, None, error_msg
                
            # Show detailed error message for original error
            import traceback
            error_msg = f"Error executing SQL query: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            return False, None, error_msg
    
    def _extract_sql_query(self, agent_result: Dict[str, Any]) -> Optional[str]:
        """
        Extract the SQL query from the agent's response.
        
        Args:
            agent_result: Result from LangChain's SQL agent
            
        Returns:
            Optional[str]: Extracted SQL query or None
        """
        # Try to extract SQL query from agent's response
        try:
            # Check if the result contains intermediate steps
            if 'intermediate_steps' in agent_result:
                for step in agent_result['intermediate_steps']:
                    # Look for SQL query in the step
                    if isinstance(step, tuple) and len(step) >= 2:
                        action, observation = step
                        if hasattr(action, 'tool') and action.tool == 'sql_db_query':
                            return action.tool_input.get('query')
            
            # If we couldn't find the SQL query in intermediate steps,
            # try to extract it from the output text
            output = agent_result.get('output', '')
            if isinstance(output, str) and 'SQL Query:' in output:
                # Extract SQL query from output text
                sql_start = output.find('SQL Query:') + len('SQL Query:')
                sql_end = output.find('\n\n', sql_start)
                if sql_end == -1:
                    sql_end = len(output)
                return output[sql_start:sql_end].strip()
                
            return None
        except Exception:
            return None
    
    def _execute_raw_sql(self, sql_query: str) -> pd.DataFrame:
        """
        Execute a raw SQL query and return the results as a DataFrame.
        
        Args:
            sql_query: SQL query to execute
            
        Returns:
            pd.DataFrame: Query results
        """
        # Execute the SQL query using the database connection
        return self.db_connection.run(sql_query)
    
    def _parse_agent_response(self, agent_result: Dict[str, Any]) -> Tuple[bool, Optional[pd.DataFrame], Optional[str]]:
        """
        Parse the agent's response to extract a DataFrame.
        
        Args:
            agent_result: Result from LangChain's SQL agent
            
        Returns:
            Tuple containing:
            - bool: Success status
            - Optional[pd.DataFrame]: Query results as DataFrame (if successful)
            - Optional[str]: Error message (if unsuccessful)
        """
        try:
            # Get the output text from the agent's response
            output = agent_result.get('output', '')
            
            # Check if the output contains a table
            if isinstance(output, str) and ('|' in output or 'Table:' in output):
                # Try to parse the table from the output
                return True, self._parse_table_from_text(output), None
            
            # If we couldn't parse a table, return the output as text
            return True, pd.DataFrame({'result': [output]}), None
            
        except Exception as e:
            return False, None, f"Error parsing agent response: {str(e)}"
    
    def _parse_table_from_text(self, text: str) -> pd.DataFrame:
        """
        Parse a table from text.
        
        Args:
            text: Text containing a table
            
        Returns:
            pd.DataFrame: Parsed table
        """
        # Find the table in the text
        table_start = text.find('|')
        if table_start == -1:
            table_start = text.find('Table:')
            if table_start != -1:
                table_start = text.find('\n', table_start) + 1
        
        if table_start == -1:
            # No table found, return a DataFrame with the text
            return pd.DataFrame({'result': [text]})
        
        # Extract the table text
        table_text = text[table_start:]
        
        # Split the table text into lines
        lines = table_text.strip().split('\n')
        
        # Parse the header
        header_line = lines[0]
        headers = [h.strip() for h in header_line.split('|') if h.strip()]
        
        # Skip the separator line
        data_lines = lines[2:] if len(lines) > 2 else []
        
        # Parse the data
        data = []
        for line in data_lines:
            if '|' in line:
                row = [cell.strip() for cell in line.split('|') if cell.strip()]
                if row:
                    data.append(row)
        
        # Create a DataFrame
        if data:
            # Make sure all rows have the same number of columns
            max_cols = max(len(row) for row in data)
            data = [row + [''] * (max_cols - len(row)) for row in data]
            
            # Create DataFrame with headers if available
            if headers and len(headers) == len(data[0]):
                return pd.DataFrame(data, columns=headers)
            else:
                return pd.DataFrame(data)
        else:
            # No data found, return a DataFrame with the text
            return pd.DataFrame({'result': [text]})
    
    def get_tables(self) -> List[str]:
        """
        Get a list of tables in the connected database.
        
        Returns:
            List[str]: List of table names
        """
        if not self.connected or not self.db_connection:
            return []
            
        try:
            return self.db_connection.get_table_names()
        except Exception:
            return []
    
    def get_table_info(self, table_name: str) -> str:
        """
        Get detailed information about a table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            str: Table information including columns, types, and constraints
        """
        if not self.connected or not self.db_connection:
            return "Not connected to a database"
            
        try:
            # Get basic table schema
            schema_info = self.db_connection.get_table_info([table_name])
            
            # Get column information
            try:
                column_query = f"""
                SELECT
                    column_name,
                    data_type,
                    column_default,
                    is_nullable
                FROM
                    information_schema.columns
                WHERE
                    table_name = '{table_name}'
                ORDER BY
                    ordinal_position
                """
                
                # Try to get detailed column information
                success, columns_df, _ = self.execute_direct_sql(column_query)
                if success and columns_df is not None and not columns_df.empty:
                    schema_info += f"\n\nColumn Details:\n{columns_df.to_string()}"
            except Exception as e:
                print(f"Error getting column details: {str(e)}")
                
            # Get primary key information
            try:
                pk_query = f"""
                SELECT
                    column_name
                FROM
                    information_schema.key_column_usage
                WHERE
                    table_name = '{table_name}'
                    AND constraint_name = 'PRIMARY'
                """
                success, pk_df, _ = self.execute_direct_sql(pk_query)
                if success and pk_df is not None and not pk_df.empty:
                    pk_columns = ", ".join(pk_df['column_name'].tolist())
                    schema_info += f"\n\nPrimary Key: {pk_columns}"
            except Exception as e:
                print(f"Error getting primary key info: {str(e)}")
                
            # Get foreign key information
            try:
                fk_query = f"""
                SELECT
                    column_name,
                    referenced_table_name,
                    referenced_column_name
                FROM
                    information_schema.key_column_usage
                WHERE
                    table_name = '{table_name}'
                    AND referenced_table_name IS NOT NULL
                """
                success, fk_df, _ = self.execute_direct_sql(fk_query)
                if success and fk_df is not None and not fk_df.empty:
                    schema_info += "\n\nForeign Keys:"
                    for _, row in fk_df.iterrows():
                        schema_info += f"\n{row['column_name']} -> {row['referenced_table_name']}.{row['referenced_column_name']}"
            except Exception as e:
                print(f"Error getting foreign key info: {str(e)}")
                
            return schema_info
        except Exception as e:
            return f"Error getting table info: {str(e)}"
    
    def disconnect(self) -> None:
        """
        Disconnect from the database.
        """
        self.db_connection = None
        self.agent = None
        self.connected = False
        self.connection_info = {}