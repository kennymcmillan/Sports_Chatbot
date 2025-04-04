"""
Chat UI Module for Multi-Source Julius AI Chatbot.

This module provides UI components for the chat interface.
"""

import os
import streamlit as st
from julius_api import Julius
from utils import process_julius_response

# Try to import the streamlit_chat component, but provide a fallback if it fails
try:
    from streamlit_chat import message
    STREAMLIT_CHAT_AVAILABLE = True
except Exception as e:
    print(f"Error importing streamlit_chat: {e}")
    STREAMLIT_CHAT_AVAILABLE = False

def display_chat_message(message, processed_response=None):
    """
    Display a chat message.
    
    Args:
        message: The message to display
        processed_response: The processed response (for assistant messages)
    """
    with st.chat_message(message["role"]):
        # For assistant messages, clean up any error traces before displaying
        if message["role"] == "assistant":
            # Clean up error traces from the content
            content = message["content"]
            
            # More aggressive cleaning of error traces
            # Remove JSON error traces that start with {"errors": and contain traceback information
            if '{"errors":' in content:
                # Find the start of the error JSON
                error_start = content.find('{"errors":')
                if error_start >= 0:
                    # Keep only the content before the error
                    clean_content = content[:error_start].strip()
                    
                    # Look for the end of the error trace
                    error_end = content.find('"]}}', error_start)
                    if error_end > 0 and error_end + 4 < len(content):
                        # Extract any content after the error trace
                        conclusion = content[error_end + 4:].strip()
                        if conclusion and not conclusion.startswith('"}'):
                            clean_content += " " + conclusion
                    
                    content = clean_content
            
            # Also clean up any FileNotFoundError traces
            if 'FileNotFoundError' in content:
                lines = content.split('\n')
                clean_lines = []
                skip_mode = False
                
                for line in lines:
                    if 'FileNotFoundError' in line or 'Traceback' in line:
                        skip_mode = True
                    elif skip_mode and line.strip() and not line.startswith(' '):
                        # This line doesn't look like part of the traceback
                        skip_mode = False
                        
                    if not skip_mode:
                        clean_lines.append(line)
                
                content = '\n'.join(clean_lines)
            
            # Display the cleaned content
            st.markdown(content)
        else:
            # For user messages, display as is
            st.markdown(message["content"])

        # If there's a processed response for assistant messages, display it
        if message["role"] == "assistant" and processed_response:
            # Display full text in an expander if it's long
            if processed_response.get('text') and len(processed_response['text']) > 200:
                with st.expander("View full response", expanded=False):
                    st.markdown(processed_response['text'])

def render_chat_column():
    """
    Render the chat interface column.
    
    This function displays:
    - Chat messages
    - Chat input
    - Handles sending messages to Julius API
    """
    st.header("Chat")
    
    # Create a scrollable container for chat with fixed height
    chat_container = st.container(height=600, border=True)
    
    # Chat input - place it ABOVE the messages so it stays at the top
    if STREAMLIT_CHAT_AVAILABLE:
        prompt = st.chat_input("Say something")
    else:
        prompt = st.text_input("Say something")
    
    # Display chat messages from history on app rerun
    with chat_container:
        # Clear any previous content
        st.empty()
        # Display all messages
        for i, message_content in enumerate(st.session_state['messages']):
            if message_content: # Check if message_content is not None
                display_chat_message(message_content, processed_response=st.session_state['responses'][i] if i < len(st.session_state['responses']) else None)
    
    # Process user input and get response
    if prompt:
        # Add user message to chat history
        st.session_state['messages'].append({"role": "user", "content": prompt})
        display_chat_message({"role": "user", "content": prompt})
        
        # Get response from Julius API
        try:
            # Initialize Julius client if not already in session state
            if 'julius' not in st.session_state:
                api_key_from_env = os.getenv("JULIUS_API_TOKEN")
                st.session_state['julius'] = Julius(api_key=api_key_from_env)
            
            # Get Julius client from session state
            julius = st.session_state['julius']
            
            # Create a message object for the Julius API
            messages = [{"role": "user", "content": prompt}]
            
            # Check if there's a processed dataset in temp_files
            processed_dataset_path = "temp_files/current_dataset.csv"
            if os.path.exists(processed_dataset_path) and (
                st.session_state['dataset'] is None or
                st.session_state.get('using_processed_dataset', False) == False
            ):
                try:
                    # Load the processed dataset
                    import pandas as pd
                    processed_df = pd.read_csv(processed_dataset_path)
                    # Update the session state
                    st.session_state['dataset'] = processed_df
                    st.session_state['dataset_name'] = "current_dataset.csv"
                    st.session_state['dataset_source'] = 'processed'
                    st.session_state['using_processed_dataset'] = True
                    print(f"--- Debug: Loaded processed dataset from {processed_dataset_path} ---")
                except Exception as e:
                    print(f"--- Debug: Error loading processed dataset: {e} ---")
            
            # If there's a dataset loaded, include information about it in the message
            if st.session_state['dataset'] is not None:
                try:
                    # Convert DataFrame information to strings to avoid JSON serialization issues
                    dataset_shape = f"{st.session_state['dataset'].shape[0]} rows x {st.session_state['dataset'].shape[1]} columns"
                    dataset_info = f"\n\nI have a dataset loaded: {st.session_state['dataset_name']} with shape {dataset_shape}."
                    
                    # Add a sample of the dataset to help the model understand its structure
                    dataset_sample = st.session_state['dataset'].head(5).to_string()
                    dataset_info += f"\n\nHere's a sample of the dataset:\n{dataset_sample}"
                    
                    # Get column names
                    column_names = ", ".join(st.session_state['dataset'].columns.tolist())
                    dataset_info += f"\n\nColumns: {column_names}"
                    
                    # Save the dataset to temp_files for access
                    try:
                        os.makedirs("temp_files", exist_ok=True)
                        # Use a fixed filename for consistency
                        fixed_dataset_path = "temp_files/current_dataset.csv"
                        st.session_state['dataset'].to_csv(fixed_dataset_path, index=False)
                        dataset_info += f"\n\nI've saved the dataset to: {fixed_dataset_path}"
                        print(f"--- Debug: Saved dataset to {fixed_dataset_path} ---")
                        
                        # Also save with original name for reference if it's not already the processed dataset
                        if st.session_state['dataset_name'] != "current_dataset.csv":
                            original_dataset_path = f"temp_files/{st.session_state['dataset_name']}"
                            st.session_state['dataset'].to_csv(original_dataset_path, index=False)
                            print(f"--- Debug: Also saved dataset to {original_dataset_path} ---")
                    except Exception as e:
                        print(f"--- Debug: Error saving dataset to temp_files: {e} ---")
                    
                    # Add instructions for accessing the dataset in Python code
                    dataset_info += f"\n\nIMPORTANT: When writing Python code to analyze this dataset, use this path:"
                    dataset_info += f"\n```python"
                    dataset_info += f"\nimport pandas as pd"
                    dataset_info += f"\ndf = pd.read_csv('{fixed_dataset_path}')"
                    dataset_info += f"\n```"
                    
                    # Append this information to the user's message
                    messages[0]["content"] += dataset_info
                    
                    print(f"--- Debug: Including dataset info in message: {st.session_state['dataset_name']} ---")
                except Exception as e:
                    print(f"--- Debug: Error including dataset info: {e} ---")
            
            # Call the completions.create method with the messages
            response = julius.chat.completions.create(messages)
            
            # Extract the response content
            response_content = response.message.content
            
            print(f"--- Debug: Processing response (length: {len(response_content)}) ---")
            processed_response = process_julius_response(response_content)
            print(f"--- Debug: Found {len(processed_response['tables'])} direct table matches ---")
            st.session_state['responses'].append(processed_response)
            print(f"--- Debug: Exported processed response to outputs/processed_response.json ---")
            with open("outputs/processed_response.json", 'w') as f:
                import json
                json.dump(processed_response, f, indent=4)
            print(f"--- Debug: Updating session state ---")
            st.session_state['messages'].append({"role": "assistant", "content": processed_response.get('text', "No text response")})
            print(f"--- Debug: Session state updated successfully ---")
            display_chat_message({"role": "assistant", "content": processed_response.get('text', "No text response")}, processed_response=processed_response)
        except Exception as e:
            error_message = f"Error communicating with Julius API: {e}"
            st.error(error_message)
            
            # Create a friendly message prompting the user to check the visualization container
            friendly_message = "I encountered an error processing your request. Please check the visualization container on the right for any results that might have been generated before the error occurred."
            
            # Check if there's any generated code to display
            code_files = [f for f in os.listdir("outputs") if f.startswith("generated_code")]
            if code_files:
                friendly_message += f"\n\nI've generated some code that might be helpful. You can see it in the visualization container."
            
            # Create a simple processed response with the friendly message
            processed_response = {
                'type': 'plain',
                'text': friendly_message,
                'tables': [],
                'html_tables': [],
                'image_urls': [],
                'generated_code_file': code_files[0] if code_files else None
            }
            
            # Add to session state and display
            st.session_state['responses'].append(processed_response)
            st.session_state['messages'].append({"role": "assistant", "content": friendly_message})
            display_chat_message({"role": "assistant", "content": friendly_message}, processed_response=processed_response)