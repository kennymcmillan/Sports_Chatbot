import streamlit as st
import os
import json
import re
import pandas as pd
import matplotlib.pyplot as plt
import io
import sys
from contextlib import redirect_stdout
from PIL import Image

def display_generated_code(processed_response, display_container):
    """
    Display generated code from a processed response if available.
    
    Args:
        processed_response: The processed response from Julius AI
        display_container: The Streamlit container to display the code in
    """
    if 'generated_code_file' in processed_response and processed_response['generated_code_file']:
        code_file = processed_response['generated_code_file']
        try:
            if os.path.exists(code_file):
                with open(code_file, "r") as f:
                    code_content = f.read()
                
                # Extract Python code from JSON if needed
                extracted_code = ""
                print(f"--- Debug: Raw code content: {code_content} ---")
                
                if code_content.strip().startswith('{') and '"python":' in code_content:
                    try:
                        # Try to parse as JSON
                        code_json = json.loads(code_content)
                        if 'python' in code_json:
                            extracted_code = code_json['python']
                            print(f"--- Debug: Successfully extracted code using JSON parsing ---")
                    except json.JSONDecodeError as e:
                        print(f"--- Debug: JSON parsing failed: {e} ---")
                        # If JSON parsing fails, try regex extraction
                        python_match = re.search(r'"python":\s*"(.*?)(?<!\\)"(?=\}|$)', code_content, re.DOTALL)
                        if python_match:
                            # Unescape the code
                            extracted_code = python_match.group(1)
                            extracted_code = extracted_code.replace('\\n', '\n').replace('\\"', '"')
                            print(f"--- Debug: Successfully extracted code using regex ---")
                
                if extracted_code:
                    print(f"--- Debug: Extracted code: {extracted_code[:50]}... ---")
                    # Display the code in an expander
                    with display_container.expander("Generated Python Code", expanded=True):
                        display_container.code(extracted_code, language="python")
                    print(f"--- Debug: Displayed generated code from {code_file} ---")
                    
                    # Save the extracted code as a .py file
                    py_filename = code_file.replace(".txt", ".py")
                    with open(py_filename, "w") as f:
                        f.write(extracted_code)
                    
                    # Add a message about the saved file
                    display_container.success(f"Code saved to {py_filename}")
                    
                    # Add a button to run the Python code
                    if display_container.button("Run Python Code"):
                        try:
                            # Create a temporary namespace to execute the code
                            namespace = {}
                            
                            # Capture print output
                            import io
                            import sys
                            from contextlib import redirect_stdout
                            
                            # Create a StringIO object to capture stdout
                            captured_output = io.StringIO()
                            
                            # Execute the code with stdout redirected to our StringIO object
                            with redirect_stdout(captured_output):
                                exec(extracted_code, namespace)
                            
                            # Get the captured output
                            output = captured_output.getvalue()
                            
                            # Display the results
                            if output:
                                display_container.text_area("Output", output, height=300)
                            
                            display_container.success("Code executed successfully!")
                        except Exception as e:
                            display_container.error(f"Error executing code: {e}")
                else:
                    # If extraction failed, just display the original content
                    with display_container.expander("Generated Code (Original)", expanded=True):
                        display_container.code(code_content, language="text")
                    print(f"--- Debug: Displayed original code content from {code_file} ---")
        except Exception as e:
            display_container.error(f"Error displaying generated code from {code_file}: {e}")

def display_database_reasoning_response(response, results_df, container):
    """
    Display a database reasoning response in a structured format.
    
    Args:
        response: The DatabaseReasoningResponse object
        results_df: The DataFrame containing query results
        container: The Streamlit container to display the response in
    """
    # Display the main analysis text
    analysis_text = response.analysis
    
    # Process the text to handle collapsible sections
    if "<details>" in analysis_text:
        # Split by details tags to process each section
        parts = re.split(r'(<details>.*?</details>)', analysis_text, flags=re.DOTALL)
        
        for part in parts:
            if not part.strip():
                continue
                
            if part.startswith("<details>"):
                # Extract summary and content
                summary_match = re.search(r'<summary>(.*?)</summary>', part, re.DOTALL)
                summary = summary_match.group(1) if summary_match else "Details"
                
                # Extract content
                content_match = re.search(r'</summary>\s*(.*?)\s*</details>', part, re.DOTALL)
                details_content = content_match.group(1) if content_match else part
                
                # Check if it contains code blocks
                code_match = re.search(r'```(\w*)\s*(.*?)\s*```', details_content, re.DOTALL)
                if code_match:
                    lang = code_match.group(1) or "text"
                    code = code_match.group(2)
                    with container.expander(f"**{summary}**", expanded=False):
                        container.code(code, language=lang)
                else:
                    with container.expander(f"**{summary}**", expanded=False):
                        container.markdown(details_content)
            else:
                # Regular content
                container.markdown(part)
    else:
        # No collapsible sections, display as normal
        container.markdown(analysis_text)
    
    # Display query results if available
    if results_df is not None and not results_df.empty:
        container.markdown("### Query Results")
        container.dataframe(results_df)
    
    # Display SQL query if not already in a collapsible container
    if response.sql_query and "<details>" not in analysis_text:
        with container.expander("SQL Query", expanded=False):
            container.code(response.sql_query, language="sql")
    
    # Display Python code if not already in a collapsible container
    if response.code and "<details>" not in analysis_text:
        with container.expander("Python Code", expanded=False):
            container.code(response.code, language="python")
    
    # Display images if available
    if response.image_urls:
        container.markdown("### Visualizations")
        for img_url in response.image_urls:
            try:
                if os.path.exists(img_url):
                    container.image(img_url)
                else:
                    # Try to load from URL
                    container.image(img_url)
            except Exception as e:
                container.error(f"Error displaying image: {str(e)}")