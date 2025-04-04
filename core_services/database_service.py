"""
Database Service Module for Multi-Source Julius AI Chatbot.

This module provides the DatabaseService class for handling database connections and queries.
It uses SQLAlchemy 2.0 for database connectivity and provides methods for
connecting to various database types and executing queries.
"""

import os
from typing import Dict, List, Optional, Union, Any, Tuple
from urllib.parse import quote_plus

import polars as pl
import pandas as pd
from sqlalchemy import create_engine, text, MetaData, Table, Column, inspect
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from pydantic import BaseModel, Field
import pymysql
import time

class TableSchema(BaseModel):
    """Table schema model."""
    name: str = Field(..., description="Table name")
    columns: List[Dict[str, Any]] = Field(..., description="Column information")
    primary_keys: List[str] = Field(default_factory=list, description="Primary key columns")
    foreign_keys: List[Dict[str, Any]] = Field(default_factory=list, description="Foreign key relationships")
    indexes: List[Dict[str, Any]] = Field(default_factory=list, description="Table indexes")
    row_count: Optional[int] = Field(None, description="Approximate row count")

class DatabaseService:
    """
    Service for handling database connections and queries.
    
    This service provides methods for:
    - Connecting to various database types (MySQL, PostgreSQL, SQLite, etc.)
    - Executing SQL queries
    - Getting database schema information
    - Converting query results to Polars DataFrames
    """
    
    def __init__(self):
        """Initialize the DatabaseService."""
        self.engines = {}
        self.metadata = {}
        self.inspectors = {}
        self.connections = {}
        self.direct_connections = {}  # Store direct pymysql connections
        self.current_connection = None
        self._schema_cache = {}  # Cache for table schemas
    
    def connect(self, db_type: str, host: str = "", port: int = 0, 
                user: str = "", password: str = "", database: str = "",
                connection_name: str = "default", use_env: bool = False) -> Tuple[bool, Optional[str]]:
        """
        Connect to a database.
        
        Args:
            db_type: Database type ('mysql', 'postgresql', 'sqlite')
            host: Database host
            port: Database port
            user: Database user
            password: Database password
            database: Database name
            connection_name: Name for this connection
            use_env: Use environment variables for connection
            
        Returns:
            Tuple containing:
            - bool: Success status
            - Optional[str]: Error message or None
        """
        try:
            # Use environment variables if requested
            if use_env:
                db_type = os.getenv("DB_TYPE", "mysql").lower()
                host = os.getenv("DB_HOST", "localhost")
                port = int(os.getenv("DB_PORT", "3306"))
                user = os.getenv("DB_USER", "")
                password = os.getenv("DB_PASSWORD", "")
                database = os.getenv("DB_NAME", "")
            
            # Create connection string
            if db_type == 'sqlite':
                connection_string = f"sqlite:///{database}"
            elif db_type == 'mysql':
                connection_string = f"mysql+pymysql://{user}:{quote_plus(password)}@{host}:{port}/{database}?connect_timeout=10"
            elif db_type == 'postgresql':
                connection_string = f"postgresql+psycopg2://{user}:{quote_plus(password)}@{host}:{port}/{database}"
            else:
                return False, f"Unsupported database type: {db_type}"
            
            # Create SQLAlchemy engine (synchronous only)
            engine = create_engine(connection_string)
            
            # Create direct connection for MySQL
            direct_connection = None
            if db_type == 'mysql':
                try:
                    # Create a direct pymysql connection with increased timeouts
                    direct_connection = pymysql.connect(
                        host=host,
                        port=port,
                        user=user,
                        password=password,
                        database=database,
                        ssl_disabled=True,  # Disable SSL for testing
                        connect_timeout=30,  # Increased from 10
                        read_timeout=120,    # Increased from 30
                        write_timeout=60     # Increased from 30
                    )
                    print(f"Successfully created direct pymysql connection to {host}:{port}/{database}")
                except Exception as e:
                    print(f"Warning: Could not create direct pymysql connection: {str(e)}")
            
            # Test connection
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            # Store connection info
            self.engines[connection_name] = engine
            self.metadata[connection_name] = MetaData()
            self.inspectors[connection_name] = inspect(engine)
            self.direct_connections[connection_name] = direct_connection
            self.connections[connection_name] = {
                'type': db_type,
                'host': host,
                'port': port,
                'user': user,
                'database': database,
                'connection_string': connection_string
            }
            
            # Set as current connection if first one
            if self.current_connection is None:
                self.current_connection = connection_name
            
            # Clear schema cache for this connection
            self._clear_schema_cache(connection_name)
            
            return True, None
        except Exception as e:
            return False, str(e)
    
    def _clear_schema_cache(self, connection_name: Optional[str] = None):
        """
        Clear the schema cache for a specific connection or all connections.
        
        Args:
            connection_name: Name of the connection (clears all if None)
        """
        if connection_name:
            # Clear cache for specific connection
            keys_to_remove = [k for k in self._schema_cache.keys() if k.startswith(f"{connection_name}_")]
            for key in keys_to_remove:
                if key in self._schema_cache:
                    del self._schema_cache[key]
        else:
            # Clear all cache
            self._schema_cache = {}
    
    def disconnect(self, connection_name: Optional[str] = None) -> bool:
        """
        Disconnect from a database.
        
        Args:
            connection_name: Name of the connection to disconnect (uses current if None)
            
        Returns:
            bool: Success status
        """
        if connection_name is None:
            connection_name = self.current_connection
        
        if connection_name not in self.engines:
            return False
        
        try:
            # Dispose engine
            self.engines[connection_name].dispose()
            
            # Remove connection info
            del self.engines[connection_name]
            del self.metadata[connection_name]
            del self.inspectors[connection_name]
            del self.connections[connection_name]
            
            # Update current connection
            if self.current_connection == connection_name:
                self.current_connection = next(iter(self.connections)) if self.connections else None
            
            return True
        except Exception:
            return False
    
    def get_connections(self) -> List[str]:
        """
        Get list of available connections.
        
        Returns:
            List[str]: List of connection names
        """
        return list(self.connections.keys())
    
    def get_tables(self, connection_name: Optional[str] = None) -> List[str]:
        """
        Get list of tables in the database.
        
        Args:
            connection_name: Name of the connection (uses current if None)
            
        Returns:
            List[str]: List of table names
        """
        if connection_name is None:
            connection_name = self.current_connection
        
        if connection_name not in self.inspectors:
            return []
        
        try:
            return self.inspectors[connection_name].get_table_names()
        except Exception:
            return []
    
    def get_table_schema(self, table_name: str, connection_name: Optional[str] = None) -> Optional[TableSchema]:
        """
        Get schema information for a table.
        
        Args:
            table_name: Name of the table
            connection_name: Name of the connection (uses current if None)
            
        Returns:
            Optional[TableSchema]: Table schema information or None
        """
        if connection_name is None:
            connection_name = self.current_connection
        
        if connection_name not in self.inspectors:
            return None
        
        # Check if we have a cached schema for this table
        cache_key = f"{connection_name}_{table_name}"
        if hasattr(self, '_schema_cache') and cache_key in self._schema_cache:
            return self._schema_cache[cache_key]
        
        try:
            inspector = self.inspectors[connection_name]
            
            # Get column information
            columns = []
            for column in inspector.get_columns(table_name):
                columns.append({
                    'name': column['name'],
                    'type': str(column['type']),
                    'nullable': column.get('nullable', True),
                    'default': str(column.get('default', '')) if column.get('default') is not None else None,
                    'autoincrement': column.get('autoincrement', False)
                })
            
            # Get primary key information
            primary_keys = []
            try:
                pk_constraint = inspector.get_pk_constraint(table_name)
                if pk_constraint and 'constrained_columns' in pk_constraint:
                    primary_keys = pk_constraint['constrained_columns']
            except Exception as e:
                # Log the error but continue - primary keys are not critical
                print(f"Error getting primary key info: {str(e)}")
            
            # Get foreign key information
            foreign_keys = []
            try:
                for fk in inspector.get_foreign_keys(table_name):
                    foreign_keys.append({
                        'name': fk.get('name'),
                        'referred_table': fk.get('referred_table'),
                        'referred_columns': fk.get('referred_columns', []),
                        'constrained_columns': fk.get('constrained_columns', [])
                    })
            except Exception:
                pass
            
            # Get index information - only if needed
            indexes = []
            
            # Skip row count for better performance - it's expensive for large tables
            # and rarely needed for schema information
            row_count = None
            
            schema = TableSchema(
                name=table_name,
                columns=columns,
                primary_keys=primary_keys,
                foreign_keys=foreign_keys,
                indexes=indexes,
                row_count=row_count
            )
            
            # Cache the schema
            if not hasattr(self, '_schema_cache'):
                self._schema_cache = {}
            self._schema_cache[cache_key] = schema
            
            return schema
        except Exception as e:
            print(f"Error getting table schema: {str(e)}")
            return None
    
    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None,
                     connection_name: Optional[str] = None, timeout: int = 60) -> Tuple[bool, Optional[pl.DataFrame], Optional[str]]:
        """
        Execute a SQL query.
        
        Args:
            query: SQL query to execute
            params: Query parameters
            connection_name: Name of the connection (uses current if None)
            timeout: Query timeout in seconds (default: 60)
            
        Returns:
            Tuple containing:
            - bool: Success status
            - Optional[pl.DataFrame]: Result dataframe or None
            - Optional[str]: Error message or None
        """
        if connection_name is None:
            connection_name = self.current_connection
        
        if connection_name not in self.engines:
            return False, None, "Database connection not found"
        
        # Clean up the SQL query to ensure it's valid
        # Remove line breaks and extra whitespace
        import re
        query = ' '.join(query.split())
        
        # Fix common syntax issues with MySQL
        # Replace multiple commas with single commas
        query = re.sub(r',\s+,', ',', query)
        
        # Remove any trailing commas in column lists
        query = re.sub(r',\s*FROM', ' FROM', query, flags=re.IGNORECASE)
        
        # Add LIMIT clause if not present to prevent large result sets
        if "SELECT" in query.upper() and "LIMIT" not in query.upper():
            # Check if there's a semicolon at the end
            if query.strip().endswith(";"):
                query = query[:-1] + " LIMIT 1000;"
            else:
                query = query + " LIMIT 1000;"
        
        # Print the query for debugging
        print(f"Executing SQL query: {query}")
        
        # First try: Use direct pymysql connection
        direct_conn = self.direct_connections.get(connection_name)
        if direct_conn:
            try:
                # Reconnect if needed
                if not direct_conn.open:
                    direct_conn.connect()
                
                # Clean up the SQL query again to be sure
                query = ' '.join(query.split())
                
                start_time = time.time()
                # Execute query
                cursor = direct_conn.cursor()
                cursor.execute(query)
                
                # Fetch results
                rows = cursor.fetchall()
                
                # Get column names
                columns = [column[0] for column in cursor.description]
                
                # Create pandas DataFrame
                df = pd.DataFrame(rows, columns=columns)
                
                # Close cursor
                cursor.close()
                
                print(f"Query executed in {time.time() - start_time:.2f} seconds using direct pymysql")
                
                # Convert pandas DataFrame to polars DataFrame
                pl_df = pl.from_pandas(df)
                return True, pl_df, None
            except Exception as direct_error:
                print(f"Direct pymysql query failed: {str(direct_error)}")
                
                # Fallback to SQLAlchemy if direct connection fails
                try:
                    start_time = time.time()
                    # Use parameterized query to avoid formatting issues
                    with self.engines[connection_name].connect() as conn:
                        # Use text() to create a proper SQL expression
                        result = conn.execute(text(query), params or {})
                        df = pd.DataFrame(result.fetchall(), columns=result.keys())
                    
                    print(f"Query executed in {time.time() - start_time:.2f} seconds using SQLAlchemy")
                    
                    # Convert pandas DataFrame to polars DataFrame
                    pl_df = pl.from_pandas(df)
                    return True, pl_df, None
                except Exception as e:
                    print(f"SQLAlchemy query failed: {str(e)}")
                    return False, None, f"Error executing query: {str(e)}"
        else:
            # Try SQLAlchemy directly if no direct connection
            try:
                start_time = time.time()
                with self.engines[connection_name].connect() as conn:
                    # Use text() to create a proper SQL expression
                    result = conn.execute(text(query), params or {})
                    df = pd.DataFrame(result.fetchall(), columns=result.keys())
                
                print(f"Query executed in {time.time() - start_time:.2f} seconds using SQLAlchemy")
                
                # Convert pandas DataFrame to polars DataFrame
                pl_df = pl.from_pandas(df)
                return True, pl_df, None
            except Exception as e:
                print(f"SQLAlchemy query failed: {str(e)}")
                return False, None, f"Error executing query: {str(e)}"
    
    # Removed async query execution method
    
    def get_database_summary(self, connection_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get a summary of the database.
        """
        if connection_name is None:
            connection_name = self.current_connection
        
        if connection_name not in self.connections:
            return {}
        
        try:
            # Get connection info
            connection_info = self.connections[connection_name].copy()
            
            # Get tables
            tables = self.get_tables(connection_name)
            
            # Get table row counts
            table_info = []
            for table in tables:
                schema = self.get_table_schema(table, connection_name)
                if schema:
                    table_info.append({
                        'name': table,
                        'columns': len(schema.columns),
                        'row_count': schema.row_count
                    })
            
            return {
                'connection': connection_info,
                'tables': table_info,
                'table_count': len(tables)
            }
        except Exception as e:
            return {'error': str(e)}
    
    def get_dialect(self, connection_name: Optional[str] = None) -> str:
        """
        Get the SQL dialect for the current or specified connection.
        
        Args:
            connection_name: Name of the connection (uses current if None)
            
        Returns:
            str: SQL dialect ('mysql', 'postgresql', or 'sqlite')
        """
        if connection_name is None:
            connection_name = self.current_connection
            
        if connection_name not in self.connections:
            return "mysql"  # Default to MySQL if no connection
            
        return self.connections[connection_name]['type']