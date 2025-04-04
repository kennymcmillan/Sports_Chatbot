"""
Database Reasoning Adapter Module for Multi-Source Julius AI Chatbot.

This module provides an adapter to use the original DatabaseReasoning class
with the new application architecture.
"""

import os
import streamlit as st
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import re
import time

from data_sources.db_reasoning import DatabaseReasoning
from data_sources.sql_connector_adapter import SQLConnectorAdapter
from core_services.database_service import DatabaseService
from julius_api import Julius
from ui_components.database_reasoning_ui import render_database_reasoning_ui

def initialize_db_reasoning(database_service: DatabaseService, julius: Julius) -> DatabaseReasoning:
    """
    Initialize the DatabaseReasoning instance.
    
    Args:
        database_service: DatabaseService instance
        julius: Julius API client instance
        
    Returns:
        DatabaseReasoning: Initialized DatabaseReasoning instance
    """
    # Create SQLConnectorAdapter to adapt DatabaseService to SQLConnector interface
    sql_connector_adapter = SQLConnectorAdapter(database_service)
    
    # Initialize DatabaseReasoning if not already in session state
    if 'db_reasoning' not in st.session_state:
        st.session_state['db_reasoning'] = DatabaseReasoning(
            sql_connector=sql_connector_adapter,
            julius=julius
        )
    
    return st.session_state['db_reasoning']

def render_database_reasoning_ui_adapter(database_service, ai_service, data_service, export_service, julius_client):
    """
    Adapter function to render the database reasoning UI using the original DatabaseReasoning class.
    
    Args:
        database_service: DatabaseService instance
        ai_service: AIService instance
        data_service: DataService instance
        export_service: ExportService instance
        julius_client: Julius API client instance
    """
    # Check if julius_client is available
    if not julius_client:
        st.error("Julius API client not available. Please check your API key.")
        return
    
    # Initialize database reasoning if not already in session state
    if 'db_reasoning' not in st.session_state:
        try:
            # Create SQLConnectorAdapter to adapt DatabaseService to SQLConnector interface
            sql_connector_adapter = SQLConnectorAdapter(database_service)
            
            st.session_state['db_reasoning'] = DatabaseReasoning(
                sql_connector=sql_connector_adapter,
                julius=julius_client
            )
        except Exception as e:
            st.error(f"Failed to initialize database reasoning: {str(e)}")
            return
    
    # Get the database reasoning instance
    db_reasoning = st.session_state['db_reasoning']
    
    # Initialize mode if not set
    if 'reasoning_mode' not in st.session_state:
        st.session_state['reasoning_mode'] = "simple"
    
    # Set advanced reasoning based on mode
    db_reasoning.julius.set_advanced_reasoning(st.session_state['reasoning_mode'] == "advanced")
    
    # Render the database reasoning UI
    render_database_reasoning_ui(database_service, ai_service, data_service, export_service)