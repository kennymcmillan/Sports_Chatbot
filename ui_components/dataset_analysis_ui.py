"""
Dataset Analysis UI Module for Multi-Source Julius AI Chatbot.

This module provides UI components for analyzing the current dataset using Julius AI.
"""

import os
import streamlit as st
import pandas as pd
import time
from julius_api import Julius

def render_dataset_analysis_sidebar():
    """
    Render the dataset analysis UI in the sidebar.
    
    This function displays UI elements for:
    - Viewing current dataset information
    - Setting analysis options
    """
    st.markdown("### Dataset Analysis")
    
    # Check if a dataset is loaded
    if not st.session_state.get('dataset_loaded', False) or st.session_state.get('dataset') is None:
        st.warning("No dataset loaded. Please load a dataset first.")
        return
    
    # Display current dataset info
    st.markdown("#### Current Dataset")
    st.write(f"Name: {st.session_state.get('dataset_name', 'Unknown')}")
    st.write(f"Source: {st.session_state.get('dataset_source', 'Unknown')}")
    st.write(f"Shape: {st.session_state.get('dataset').shape}")
    
    # Dataset preview
    with st.expander("Dataset Preview", expanded=False):
        st.dataframe(st.session_state.get('dataset').head(10))
    
    # Dataset statistics
    with st.expander("Dataset Statistics", expanded=False):
        # Get numeric columns
        df = st.session_state.get('dataset')
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if numeric_cols:
            st.write("Numeric Columns Statistics:")
            st.dataframe(df[numeric_cols].describe())
        
        # Get categorical columns
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if cat_cols:
            st.write("Categorical Columns:")
            for col in cat_cols:
                st.write(f"**{col}** - Unique Values: {df[col].nunique()}")
                if df[col].nunique() < 10:  # Only show value counts for columns with few unique values
                    st.write(df[col].value_counts().head(5))
    
    # Analysis options
    st.markdown("#### Analysis Options")
    
    # Option to generate visualizations
    st.checkbox("Generate visualizations", value=True, key="generate_visualizations")
    
    # Option to perform statistical analysis
    st.checkbox("Perform statistical analysis", value=True, key="perform_statistical_analysis")
    
    # Option to generate insights
    st.checkbox("Generate insights", value=True, key="generate_insights")

def render_dataset_analysis_chat():
    """
    Render the dataset analysis chat interface.
    
    This function displays:
    - Chat history
    - Chat input
    - Analysis results
    - Generated code and visualizations
    """
    st.header("Dataset Analysis Chat")
    
    # Check if a dataset is loaded
    if not st.session_state.get('dataset_loaded', False) or st.session_state.get('dataset') is None:
        st.warning("No dataset loaded. Please load a dataset first.")
        return
    
    # Initialize Julius client if not already in session state
    if 'julius' not in st.session_state:
        api_key_from_env = os.getenv("JULIUS_API_TOKEN")
        st.session_state['julius'] = Julius(api_key=api_key_from_env)
    
    # Get Julius client from session state
    julius = st.session_state['julius']
    
    # Initialize chat history in session state
    if 'dataset_analysis_messages' not in st.session_state:
        st.session_state['dataset_analysis_messages'] = []
    
    # Initialize analysis results in session state
    if 'dataset_analysis_results' not in st.session_state:
        st.session_state['dataset_analysis_results'] = []
    
    # Create a scrollable container for chat with fixed height
    chat_container = st.container(height=400, border=True)
    
    # Display chat history
    with chat_container:
        # Clear any previous content
        st.empty()
        
        # Display all messages
        for i, message in enumerate(st.session_state['dataset_analysis_messages']):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # If it's an assistant message with analysis results, display them
                if message["role"] == "assistant" and i < len(st.session_state['dataset_analysis_results']):
                    result = st.session_state['dataset_analysis_results'][i]
                    
                    # Display generated code if available
                    if "generated_code" in result and result["generated_code"]:
                        with st.expander("Generated Python Code", expanded=False):
                            st.code(result["generated_code"], language="python")
                            
                            # Option to download code
                            if "code_filename" in result and result["code_filename"]:
                                with open(result["code_filename"], "r") as f:
                                    code_content = f.read()
                                    
                                st.download_button(
                                    label="Download Code",
                                    data=code_content,
                                    file_name=os.path.basename(result["code_filename"]),
                                    mime="text/plain"
                                )
                            
                            # Option to execute code
                            if st.button("Execute Code", key=f"execute_code_{i}"):
                                with st.spinner("Executing code..."):
                                    try:
                                        # Create a temporary Python file
                                        temp_py = f"outputs/temp_script_{int(time.time())}.py"
                                        with open(temp_py, "w") as f:
                                            f.write(result["generated_code"])
                                        
                                        # Execute the code and capture output
                                        import subprocess
                                        import sys
                                        
                                        result_exec = subprocess.run(
                                            [sys.executable, temp_py],
                                            capture_output=True,
                                            text=True
                                        )
                                        
                                        # Display output
                                        if result_exec.stdout:
                                            st.subheader("Code Output:")
                                            st.text(result_exec.stdout)
                                        
                                        # Display errors
                                        if result_exec.stderr:
                                            st.error("Code Execution Error:")
                                            st.code(result_exec.stderr)
                                        
                                        # Check for generated images
                                        import glob
                                        image_files = glob.glob("*.png") + glob.glob("*.jpg")
                                        
                                        if image_files:
                                            st.subheader("Generated Visualizations:")
                                            for img_file in image_files:
                                                st.image(img_file)
                                    except Exception as e:
                                        st.error(f"Error executing code: {str(e)}")
                    
                    # Display visualizations if available
                    if "visualizations" in result and result["visualizations"]:
                        st.subheader("Visualizations")
                        for viz in result["visualizations"]:
                            st.image(viz)
    
    # Chat input
    prompt = st.chat_input("Ask a question about your dataset...")
    
    if prompt:
        # Add user message to chat history
        st.session_state['dataset_analysis_messages'].append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Analyze the dataset
        with st.spinner("Analyzing dataset and generating response..."):
            # Get dataset information
            df = st.session_state.get('dataset')
            dataset_name = st.session_state.get('dataset_name', 'Unknown')
            dataset_source = st.session_state.get('dataset_source', 'Unknown')
            
            # Create dataset info string
            dataset_info = f"""
            Dataset Name: {dataset_name}
            Dataset Source: {dataset_source}
            Shape: {df.shape[0]} rows x {df.shape[1]} columns
            
            Columns: {', '.join(df.columns.tolist())}
            
            Sample Data:
            {df.head(5).to_string()}
            
            Data Types:
            {df.dtypes.to_string()}
            """
            
            # Get analysis options
            generate_visualizations = st.session_state.get('generate_visualizations', True)
            perform_statistical_analysis = st.session_state.get('perform_statistical_analysis', True)
            generate_insights = st.session_state.get('generate_insights', True)
            
            # Create prompt for Julius
            prompt_text = f"""
            I want you to help me analyze a dataset. Here's the dataset information:
            
            {dataset_info}
            
            User Query: {prompt}
            
            Please analyze this dataset and help me by:
            1. Explaining what information the user is looking for
            2. {f"Generating appropriate visualizations to answer the query" if generate_visualizations else ""}
            3. {f"Performing statistical analysis relevant to the query" if perform_statistical_analysis else ""}
            4. {f"Providing insights based on the data" if generate_insights else ""}
            5. Generating Python code to perform this analysis
            
            For the Python code:
            - Use pandas, matplotlib, and seaborn libraries
            - Make the code complete and executable
            - Include code to load the dataset from: "temp_files/current_dataset.csv"
            - Format your code using markdown code blocks with the python language specifier
            """
            
            # Send query to Julius
            messages = [{"role": "user", "content": prompt_text}]
            response = julius.chat.completions.create(messages)
            
            # Extract response content
            response_content = response.message.content
            
            # Extract code from the response
            import re
            code_pattern = r'```python\s*(.*?)\s*```'
            code_matches = re.findall(code_pattern, response_content, re.DOTALL)
            
            generated_code = None
            code_filename = None
            
            if code_matches:
                generated_code = code_matches[0]
                
                # Save the code to a file
                os.makedirs("outputs", exist_ok=True)
                code_filename = f"outputs/analysis_code_{int(time.time())}.py"
                with open(code_filename, "w") as f:
                    f.write(generated_code)
                
                print(f"Generated code saved to {code_filename}")
            
            # Create analysis result
            analysis_result = {
                "text": response_content,
                "generated_code": generated_code,
                "code_filename": code_filename,
                "visualizations": []  # Will be populated when code is executed
            }
            
            # Add assistant message to chat history
            st.session_state['dataset_analysis_messages'].append({
                "role": "assistant", 
                "content": response_content
            })
            
            # Add analysis result to results history
            st.session_state['dataset_analysis_results'].append(analysis_result)
            
            # Display assistant message
            with st.chat_message("assistant"):
                st.markdown(response_content)
                
                # Display generated code if available
                if generated_code:
                    with st.expander("Generated Python Code", expanded=False):
                        st.code(generated_code, language="python")
                        
                        # Option to download code
                        if code_filename:
                            with open(code_filename, "r") as f:
                                code_content = f.read()
                                
                            st.download_button(
                                label="Download Code",
                                data=code_content,
                                file_name=os.path.basename(code_filename),
                                mime="text/plain"
                            )
                        
                        # Option to execute code
                        if st.button("Execute Code"):
                            with st.spinner("Executing code..."):
                                try:
                                    # Create a temporary Python file
                                    temp_py = f"outputs/temp_script_{int(time.time())}.py"
                                    with open(temp_py, "w") as f:
                                        f.write(generated_code)
                                    
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
                                    image_files = glob.glob("*.png") + glob.glob("*.jpg")
                                    
                                    if image_files:
                                        st.subheader("Generated Visualizations:")
                                        for img_file in image_files:
                                            st.image(img_file)
                                except Exception as e:
                                    st.error(f"Error executing code: {str(e)}")