"""
SQL Connector Adapter Module for Multi-Source Julius AI Chatbot.

This module provides an adapter that makes DatabaseService compatible with the SQLConnector interface
expected by the DatabaseReasoning class, implementing a hybrid architecture approach.
"""

import os
import pandas as pd
import polars as pl
from typing import Dict, List, Optional, Any, Tuple
import streamlit as st

from core_services.database_service import DatabaseService

class SQLConnectorAdapter:
    """
    Adapter class that makes DatabaseService compatible with the SQLConnector interface.
    
    This class wraps a DatabaseService instance and provides methods that match
    the SQLConnector interface, allowing the DatabaseReasoning class to work with
    either implementation.
    """
    
    def __init__(self, database_service: DatabaseService):
        """
        Initialize the SQL Connector Adapter.
        
        Args:
            database_service: DatabaseService instance to adapt
        """
        self.database_service = database_service
        self.connected = database_service.current_connection is not None
        self.connection_info = {}
        
        # Copy connection info if available
        if self.connected and database_service.current_connection:
            self.connection_info = database_service.connections.get(database_service.current_connection, {})
    
    def connect_to_database(self,
                           db_type: str = None,
                           host: str = None,
                           user: str = None,
                           password: str = None,
                           database: str = None,
                           port: Optional[int] = None,
                           use_env: bool = False) -> bool:
        """
        Connect to a SQL database using the DatabaseService.
        
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
        success, error = self.database_service.connect(
            db_type=db_type,
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            use_env=use_env
        )
        
        self.connected = success
        
        # Update connection info if successful
        if success and self.database_service.current_connection:
            self.connection_info = self.database_service.connections.get(self.database_service.current_connection, {})
        
        return success
    
    def execute_query(self, query: str) -> Tuple[bool, Optional[pd.DataFrame], Optional[str]]:
        """
        Execute a natural language query using DatabaseService.
        
        Args:
            query: Natural language query
            
        Returns:
            Tuple containing:
            - bool: Success status
            - Optional[pd.DataFrame]: Query results as DataFrame (if successful)
            - Optional[str]: Error message (if unsuccessful)
        """
        # For now, just pass through to execute_direct_sql
        # In a more complete implementation, this would use AI to generate SQL
        return self.execute_direct_sql(query)
    
    def execute_direct_sql(self, sql_query: str) -> Tuple[bool, Optional[pd.DataFrame], Optional[str]]:
        """
        Execute a direct SQL query using DatabaseService.
        
        Args:
            sql_query: SQL query to execute
            
        Returns:
            Tuple containing:
            - bool: Success status
            - Optional[pd.DataFrame]: Query results as DataFrame (if successful)
            - Optional[str]: Error message (if unsuccessful)
        """
        if not self.connected:
            return False, None, "Not connected to a database"
        
        success, result_df, error = self.database_service.execute_query(sql_query)
        
        if success and result_df is not None:
            # Convert polars DataFrame to pandas DataFrame
            pandas_df = result_df.to_pandas()
            return True, pandas_df, None
        else:
            return False, None, error
    
    def get_tables(self) -> List[str]:
        """
        Get a list of tables in the connected database.
        
        Returns:
            List[str]: List of table names
        """
        if not self.connected:
            return []
        
        return self.database_service.get_tables()
    
    def get_table_info(self, table_name: str) -> str:
        """
        Get detailed information about a table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            str: Table information including columns, types, and constraints
        """
        if not self.connected:
            return "Not connected to a database"
        
        schema = self.database_service.get_table_schema(table_name)
        if not schema:
            return f"Could not get schema for table {table_name}"
        
        # Format schema as string (similar to SQLConnector.get_table_info)
        schema_info = f"Table: {schema.name}\nColumns:\n"
        
        for col in schema.columns:
            nullable = "NULL" if col.get('nullable', True) else "NOT NULL"
            default = f"DEFAULT {col.get('default')}" if col.get('default') is not None else ""
            schema_info += f"- {col['name']} {col['type']} {nullable} {default}\n"
        
        # Add primary key information
        if schema.primary_keys:
            schema_info += f"\nPrimary Key: {', '.join(schema.primary_keys)}"
        
        # Add foreign key information
        if schema.foreign_keys:
            schema_info += "\nForeign Keys:"
            for fk in schema.foreign_keys:
                schema_info += f"\n{', '.join(fk['constrained_columns'])} -> {fk['referred_table']}({', '.join(fk['referred_columns'])})"
        
        return schema_info
    
    def disconnect(self) -> None:
        """
        Disconnect from the database.
        """
        if self.database_service.current_connection:
            self.database_service.disconnect(self.database_service.current_connection)
            self.connected = False
            self.connection_info = {}