"""
LangChain UI Module for Multi-Source Julius AI Chatbot.

This module provides UI components for LangChain chat interface.
"""

import os
import streamlit as st
import pandas as pd
from typing import List, Dict, Any, Optional

# Try to import LangChain components
try:
    from langchain.chains import ConversationChain
    from langchain.memory import ConversationBufferMemory
    from langchain_openai import ChatOpenAI
    from langchain_community.utilities import SQLDatabase
    
    # Try to import HuggingFaceHub for free models
    try:
        # First try the standard import
        import langchain_huggingface
        from langchain_huggingface import HuggingFaceHub
        HUGGINGFACE_AVAILABLE = True
        print("HuggingFace Hub integration is available.")
    except ImportError as e:
        # If standard import fails, try alternative import paths
        try:
            # Try importing directly from langchain
            from langchain.llms import HuggingFaceHub
            HUGGINGFACE_AVAILABLE = True
            print("HuggingFace Hub integration is available (legacy import).")
        except ImportError:
            # If all imports fail, set flag to False
            HUGGINGFACE_AVAILABLE = False
            print(f"Error importing HuggingFace Hub: {e}")
            # Print more detailed error information
            import sys
            import traceback
            print("Detailed error information:")
            traceback.print_exc(file=sys.stdout)
    
    # Try to import OpenRouter integration
    try:
        from langchain_openai import ChatOpenAI
        OPENROUTER_AVAILABLE = True
    except ImportError:
        OPENROUTER_AVAILABLE = False
    
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    HUGGINGFACE_AVAILABLE = False
    OPENROUTER_AVAILABLE = False

def render_langchain_sidebar():
    """
    Render the LangChain chat sidebar.
    
    This function displays UI elements for:
    - LangChain model selection
    - Memory settings
    - Database connection settings
    """
    st.markdown("### LangChain Chat Settings")
    
    # Check if LangChain is available
    if not LANGCHAIN_AVAILABLE:
        st.error("LangChain is not installed. Please install it to use this feature.")
        st.code("pip install langchain langchain-openai langchain-community", language="bash")
        return
    
    # OpenAI API key input
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        openai_api_key = st.text_input("OpenAI API Key", type="password")
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
    
    # Model provider selection
    model_provider = st.radio(
        "Select Model Provider",
        ["OpenAI", "HuggingFace Hub (Free)", "OpenRouter (Free Models)"],
        index=2  # Default to OpenRouter as it has free models
    )
    
    if model_provider == "OpenAI":
        # OpenAI model selection
        model = st.selectbox(
            "Select OpenAI Model",
            ["gpt-4o", "gpt-3.5-turbo", "gpt-4-turbo"],
            index=1  # Default to gpt-3.5-turbo as it's cheaper
        )
    elif model_provider == "HuggingFace Hub (Free)":
        # HuggingFace model selection
        if HUGGINGFACE_AVAILABLE:
            model = st.selectbox(
                "Select HuggingFace Model",
                ["google/flan-t5-large", "google/flan-t5-xl", "google/flan-t5-xxl", "google/flan-ul2"],
                index=0  # Default to flan-t5-large as it's smaller
            )
            
            # HuggingFace API token
            huggingface_api_token = os.getenv("HUGGINGFACE_API_TOKEN")
            if not huggingface_api_token:
                huggingface_api_token = st.text_input("HuggingFace API Token (optional)", type="password")
                if huggingface_api_token:
                    os.environ["HUGGINGFACE_API_TOKEN"] = huggingface_api_token
        else:
            st.warning("HuggingFace Hub integration is not installed. Please install it with: `pip install langchain-huggingface`")
            model = "google/flan-t5-large"  # Default model even if not available
    else:  # OpenRouter
        # OpenRouter model selection
        model = st.selectbox(
            "Select OpenRouter Model",
            [
                "deepseek-ai/deepseek-coder-33b-instruct",
                "deepseek-ai/deepseek-llm-67b-chat",
                "mistralai/mistral-7b-instruct",
                "01-ai/yi-34b-chat",
                "meta-llama/llama-2-13b-chat"
            ],
            index=0  # Default to deepseek-coder
        )
        
        # OpenRouter API key
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        if not openrouter_api_key:
            openrouter_api_key = st.text_input("OpenRouter API Key", type="password")
            if openrouter_api_key:
                os.environ["OPENROUTER_API_KEY"] = openrouter_api_key
    
    # Temperature setting
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
    
    # Save settings to session state
    if 'langchain_settings' not in st.session_state:
        st.session_state['langchain_settings'] = {}
    
    st.session_state['langchain_settings']['model_provider'] = model_provider
    st.session_state['langchain_settings']['model'] = model
    st.session_state['langchain_settings']['temperature'] = temperature
    st.session_state['langchain_settings']['api_key'] = openai_api_key
    
    # Database connection settings
    with st.expander("Database Connection", expanded=False):
        # Database type selection
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
        
        # Save database settings to session state
        if 'langchain_db_settings' not in st.session_state:
            st.session_state['langchain_db_settings'] = {}
        
        st.session_state['langchain_db_settings']['db_type'] = db_type
        st.session_state['langchain_db_settings']['host'] = host
        st.session_state['langchain_db_settings']['port'] = port
        st.session_state['langchain_db_settings']['database'] = database
        st.session_state['langchain_db_settings']['user'] = user
        st.session_state['langchain_db_settings']['password'] = password
        
        # Connect button
        if st.button("Connect to Database"):
            if connect_to_database():
                st.success(f"Connected to {db_type} database: {database}")
            else:
                st.error("Failed to connect to database.")

def connect_to_database() -> bool:
    """
    Connect to the database using LangChain.
    
    Returns:
        bool: True if connection successful, False otherwise
    """
    if not LANGCHAIN_AVAILABLE:
        return False
    
    try:
        # Get database settings from session state
        db_settings = st.session_state.get('langchain_db_settings', {})
        db_type = db_settings.get('db_type', '')
        host = db_settings.get('host', '')
        port = db_settings.get('port', 0)
        database = db_settings.get('database', '')
        user = db_settings.get('user', '')
        password = db_settings.get('password', '')
        
        # Construct database URI based on database type
        if db_type.lower() == 'mysql':
            db_uri = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
        elif db_type.lower() == 'postgresql':
            db_uri = f"postgresql://{user}:{password}@{host}:{port}/{database}"
        elif db_type.lower() == 'sqlite':
            db_uri = f"sqlite:///{database}"
        else:
            st.error(f"Unsupported database type: {db_type}")
            return False
        
        # Create database connection
        db = SQLDatabase.from_uri(db_uri)
        
        # Store database connection in session state
        st.session_state['langchain_db'] = db
        st.session_state['langchain_db_connected'] = True
        
        return True
    except Exception as e:
        st.error(f"Error connecting to database: {str(e)}")
        st.session_state['langchain_db_connected'] = False
        return False

def render_langchain_chat():
    """
    Render the LangChain chat interface.
    
    This function displays:
    - Chat history
    - Chat input
    - Options to set query results as current dataset
    """
    st.header("LangChain Chat")
    
    # Check if LangChain is available
    if not LANGCHAIN_AVAILABLE:
        st.error("LangChain is not installed. Please install it to use this feature.")
        st.code("pip install langchain langchain-openai langchain-community", language="bash")
        return
    
    # Initialize LangChain chat history in session state
    if 'langchain_messages' not in st.session_state:
        st.session_state['langchain_messages'] = []
    
    # Initialize LangChain memory in session state
    if 'langchain_memory' not in st.session_state:
        st.session_state['langchain_memory'] = ConversationBufferMemory()
    
    # Initialize LangChain chain in session state
    if 'langchain_chain' not in st.session_state or st.session_state.get('langchain_chain_initialized', False) == False:
        try:
            # Get settings from session state
            settings = st.session_state.get('langchain_settings', {})
            model_provider = settings.get('model_provider', 'OpenRouter (Free Models)')
            model = settings.get('model', 'deepseek-ai/deepseek-coder-33b-instruct')
            temperature = settings.get('temperature', 0.7)
            api_key = settings.get('api_key', '')
            
            # Create LLM based on provider
            if model_provider == "OpenAI":
                llm = ChatOpenAI(
                    model=model,
                    temperature=temperature,
                    api_key=api_key
                )
            elif model_provider == "HuggingFace Hub (Free)":
                # Use HuggingFace Hub
                if HUGGINGFACE_AVAILABLE:
                    huggingface_api_token = os.getenv("HUGGINGFACE_API_TOKEN", "")
                    
                    # Check which import was successful and use the appropriate constructor
                    if 'langchain_huggingface' in sys.modules:
                        # New import style
                        llm = HuggingFaceHub(
                            repo_id=model,
                            huggingfacehub_api_token=huggingface_api_token,
                            model_kwargs={"temperature": temperature}
                        )
                    else:
                        # Legacy import style
                        llm = HuggingFaceHub(
                            repo_id=model,
                            huggingfacehub_api_token=huggingface_api_token,
                            model_kwargs={"temperature": temperature}
                        )
                else:
                    # Try to install the package on-the-fly
                    st.warning("HuggingFace Hub integration is not available. Attempting to install it now...")
                    try:
                        import subprocess
                        subprocess.check_call([sys.executable, "-m", "pip", "install", "langchain-huggingface", "huggingface_hub", "transformers"])
                        st.success("Installation successful! Please restart the application.")
                    except Exception as install_error:
                        st.error(f"HuggingFace Hub integration is not installed. Please install it manually with: `pip install langchain-huggingface huggingface_hub transformers`")
                        st.error(f"Installation error: {install_error}")
                    return
            else:  # OpenRouter
                # Use OpenRouter
                openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "")
                if not openrouter_api_key:
                    st.error("OpenRouter API Key is required. Please provide it in the sidebar.")
                    return
                
                # Configure OpenRouter base URL
                openrouter_base_url = "https://openrouter.ai/api/v1"
                
                # Use ChatOpenAI with OpenRouter configuration
                llm = ChatOpenAI(
                    model=model,
                    temperature=temperature,
                    openai_api_key=openrouter_api_key,
                    openai_api_base=openrouter_base_url
                )
            
            # Create chain
            st.session_state['langchain_chain'] = ConversationChain(
                llm=llm,
                memory=st.session_state['langchain_memory'],
                verbose=True
            )
            
            st.session_state['langchain_chain_initialized'] = True
        except Exception as e:
            st.error(f"Error initializing LangChain: {str(e)}")
            st.session_state['langchain_chain_initialized'] = False
    
    # Create a scrollable container for chat with fixed height
    chat_container = st.container(height=400, border=True)
    
    # Display chat history
    with chat_container:
        # Clear any previous content
        st.empty()
        
        # Display all messages
        for message in st.session_state['langchain_messages']:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # Chat input
    prompt = st.chat_input("Ask LangChain to retrieve data...")
    
    if prompt:
        # Add user message to chat history
        st.session_state['langchain_messages'].append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get response from LangChain
        try:
            with st.spinner("LangChain is thinking..."):
                # Check if connected to database
                if st.session_state.get('langchain_db_connected', False) and 'langchain_db' in st.session_state:
                    # Use database to answer query
                    db = st.session_state['langchain_db']
                    
                    # Create a custom prompt that includes database schema
                    db_schema = db.get_table_info()
                    custom_prompt = f"""You are an AI assistant that helps with database queries.
                    
                    Database Schema:
                    {db_schema}
                    
                    User Query: {prompt}
                    
                    Please provide:
                    1. A clear explanation of what the query is asking for
                    2. The SQL query that would answer this question
                    3. A brief explanation of the results
                    
                    If you need to generate a dataset from this query, please format your response so that it can be easily parsed and set as the current dataset.
                    """
                    
                    # Execute the query
                    response = st.session_state['langchain_chain'].predict(input=custom_prompt)
                else:
                    # Use regular conversation chain
                    response = st.session_state['langchain_chain'].predict(input=prompt)
            
            # Add assistant message to chat history
            st.session_state['langchain_messages'].append({"role": "assistant", "content": response})
            
            # Display assistant message
            with st.chat_message("assistant"):
                st.markdown(response)
            
            # Check if response contains SQL query
            if "```sql" in response.lower() or "```" in response and "select" in response.lower():
                # Extract SQL query
                sql_query = extract_sql_query(response)
                
                if sql_query and st.session_state.get('langchain_db_connected', False) and 'langchain_db' in st.session_state:
                    # Execute SQL query
                    try:
                        db = st.session_state['langchain_db']
                        result = db.run(sql_query)
                        
                        # Convert result to DataFrame
                        if isinstance(result, str):
                            # Parse result string into DataFrame
                            import io
                            df = pd.read_csv(io.StringIO(result), sep=',')
                        else:
                            # Result might already be a DataFrame or list of tuples
                            df = pd.DataFrame(result)
                        
                        # Display result
                        st.dataframe(df)
                        
                        # Option to set as current dataset
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            dataset_name = st.text_input("Dataset Name", f"langchain_query_result_{len(st.session_state['langchain_messages'])}.csv")
                        with col2:
                            if st.button("Set as Current Dataset"):
                                # Set as current dataset
                                set_as_current_dataset(df, dataset_name)
                    except Exception as e:
                        st.error(f"Error executing SQL query: {str(e)}")
        except Exception as e:
            st.error(f"Error getting response from LangChain: {str(e)}")
            
            # Add error message to chat history
            error_message = f"Error: {str(e)}"
            st.session_state['langchain_messages'].append({"role": "assistant", "content": error_message})
            
            # Display error message
            with st.chat_message("assistant"):
                st.markdown(error_message)

def extract_sql_query(text: str) -> Optional[str]:
    """
    Extract SQL query from text.
    
    Args:
        text: Text containing SQL query
        
    Returns:
        Optional[str]: Extracted SQL query or None
    """
    # Check for SQL code blocks
    if "```sql" in text.lower():
        # Extract SQL query from code block
        start = text.lower().find("```sql") + 6
        end = text.find("```", start)
        if end > start:
            return text[start:end].strip()
    
    # Check for generic code blocks that might contain SQL
    if "```" in text and "select" in text.lower():
        # Extract query from code block
        start = text.find("```") + 3
        end = text.find("```", start)
        if end > start:
            query = text[start:end].strip()
            # Verify it looks like SQL
            if "select" in query.lower():
                return query
    
    # Check for SQL queries without code blocks
    if "select" in text.lower() and "from" in text.lower():
        # Try to extract the query based on common patterns
        lines = text.split('\n')
        query_lines = []
        in_query = False
        
        for line in lines:
            if "select" in line.lower() and not in_query:
                in_query = True
                query_lines.append(line)
            elif in_query:
                query_lines.append(line)
                if ";" in line:
                    break
        
        if query_lines:
            return '\n'.join(query_lines).strip()
    
    return None

def set_as_current_dataset(df: pd.DataFrame, name: str) -> bool:
    """
    Set a dataframe as the current dataset.
    
    Args:
        df: DataFrame to set as current dataset
        name: Name for the dataset
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Update session state
        st.session_state['dataset'] = df
        st.session_state['dataset_name'] = name
        st.session_state['dataset_source'] = 'langchain'
        st.session_state['dataset_loaded'] = True
        st.session_state['cached_dataset'] = df
        
        # Save to temp_files for persistence
        os.makedirs("temp_files", exist_ok=True)
        df.to_csv("temp_files/current_dataset.csv", index=False)
        
        # Also save with original name if different
        if name != "current_dataset.csv":
            df.to_csv(f"temp_files/{name}", index=False)
        
        st.success(f"Dataset '{name}' set as current dataset.")
        
        # Switch to dataset mode
        st.session_state['current_mode'] = 'dataset'
        
        return True
    except Exception as e:
        st.error(f"Error setting dataset: {str(e)}")
        return False