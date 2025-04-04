"""
Visualization UI Module for Multi-Source Julius AI Chatbot.

This module provides UI components for visualizing results.
"""

import os
import glob
import streamlit as st
import pandas as pd
import json
import re
import requests
from io import BytesIO
from PIL import Image
from custom_display import display_generated_code

def render_visualization_column():
    """
    Render the visualization column.
    
    This function displays:
    - Tables
    - Generated code
    - Images
    - Other visualizations
    """
    st.header("Visualizations")
    
    # Create a scrollable container for visualizations with fixed height
    viz_container = st.container(height=600, border=True)
    
    with viz_container:
        # Display the last response
        if st.session_state['responses']:
            last_response = st.session_state['responses'][-1]
            display_response(last_response, viz_container)
        
        # Check for and display generated code files
        display_generated_code_files(viz_container)
        
        # Check for and display image files
        display_output_images(viz_container)

def display_response(processed_response, container=None):
    """
    Display a processed response in a structured way.
    
    Args:
        processed_response: The processed response to display
        container: The container to display visualizations in
    """
    # Use the provided container or the current context
    display_container = container if container else st

    if processed_response['type'] == 'structured':
        # Handle structured JSON response with outputs and image_urls
        data = processed_response['data']

        # Only display text in the left column, not in the visualization container
        if container:
            # Skip text display in viz container
            pass
        elif 'outputs' in data:
            # Display only text outputs in the left column
            for output in data['outputs']:
                if isinstance(output, str) and not output.startswith('[JULIUS_TABLE]:') and not output.startswith('<table'):
                    st.write(output)

        # Display visualizations in the specified container
        if 'outputs' in data:
            for output in data['outputs']:
                # Special case for outputs that start with a quote and then [JULIUS_TABLE]
                if isinstance(output, str) and output.startswith('"[JULIUS_TABLE]'):
                    print(f"--- Debug: Found output starting with quote and [JULIUS_TABLE] ---")
                    # Remove the outer quotes
                    if output.startswith('"') and output.endswith('"'):
                        output = output[1:-1]

                # Check if it's a table in JSON format
                if isinstance(output, str) and output.startswith('[JULIUS_TABLE]:'):
                    print(f"--- Debug: Found [JULIUS_TABLE] in output ---")
                    table_pattern = r'\[JULIUS_TABLE\]: "(.*?)"'
                    match = re.search(table_pattern, output)
                    if match:
                        try:
                            # Get the table JSON string and unescape it
                            table_json = match.group(1)
                            # Multiple levels of unescaping may be needed
                            while '\\\\' in table_json:
                                table_json = table_json.replace('\\\\', '\\')
                            table_json = table_json.replace('\\"', '"')

                            print(f"--- Debug: Unescaped table JSON (first 100 chars): {table_json[:100]} ---")

                            # Parse the JSON
                            table_data = json.loads(table_json)
                            df = pd.DataFrame(
                                data=table_data['data'],
                                columns=table_data.get('columns', []),
                                index=table_data.get('index', None)
                            )
                            display_container.dataframe(df)
                        except Exception as e:
                            display_container.error(f"Error displaying table: {e}")
                # Check if it's an HTML table
                elif isinstance(output, str) and output.startswith('<table'):
                    display_container.markdown(output, unsafe_allow_html=True)
                # Otherwise, display as text
                else:
                    display_container.write(output)

        # Display images
        if 'image_urls' in data and data['image_urls']:
            for url in data['image_urls']:
                try:
                    response = requests.get(url)
                    img = Image.open(BytesIO(response.content))
                    display_container.image(img, use_column_width=True)
                except Exception as e:
                    display_container.error(f"Error displaying image from {url}: {e}")

    elif processed_response['type'] == 'structured_parts':
        # Handle multiple structured parts
        for part in processed_response['data']:
            # Only display text in the left column, not in the visualization container
            if container:
                # Skip text display in viz container
                pass
            elif 'outputs' in part:
                # Display only text outputs in the left column
                for output in part['outputs']:
                    if isinstance(output, str) and not output.startswith('[JULIUS_TABLE]:') and not output.startswith('<table'):
                        st.write(output)

            # Display visualizations in the specified container
            if 'outputs' in part:
                for output in part['outputs']:
                    # Special case for outputs that start with a quote and then [JULIUS_TABLE]
                    if isinstance(output, str) and output.startswith('"[JULIUS_TABLE]'):
                        print(f"--- Debug: Found output starting with quote and [JULIUS_TABLE] in structured_parts ---")
                        # Remove the outer quotes
                        if output.startswith('"') and output.endswith('"'):
                            output = output[1:-1]

                    # Check if it's a table in JSON format
                    if isinstance(output, str) and output.startswith('[JULIUS_TABLE]:'):
                        table_pattern = r'\[JULIUS_TABLE\]: "(.*?)"'
                        match = re.search(table_pattern, output)
                        if match:
                            try:
                                # Get the table JSON string and unescape it
                                table_json = match.group(1)
                                # Multiple levels of unescaping may be needed
                                while '\\\\' in table_json:
                                    table_json = table_json.replace('\\\\', '\\')
                                table_json = table_json.replace('\\"', '"')

                                print(f"--- Debug: Unescaped table JSON in structured_parts (first 100 chars): {table_json[:100]} ---")

                                # Parse the JSON with error handling
                                try:
                                    table_data = json.loads(table_json)
                                except json.JSONDecodeError as e:
                                    print(f"--- Debug: JSON decode error in structured_parts: {e} ---")
                                    # Try to find a valid JSON substring
                                    if "Extra data" in str(e):
                                        # Find the position of the error
                                        error_pos = int(str(e).split("column ")[1].split(" ")[0])
                                        print(f"--- Debug: Error position: {error_pos} ---")
                                        # Try parsing just up to the error position
                                        table_data = json.loads(table_json[:error_pos])
                                    else:
                                        raise e

                                df = pd.DataFrame(
                                    data=table_data['data'],
                                    columns=table_data.get('columns', []),
                                    index=table_data.get('index', None)
                                )
                                display_container.dataframe(df)
                            except Exception as e:
                                display_container.error(f"Error displaying table: {e}")
                    # Check if it's an HTML table
                    elif isinstance(output, str) and output.startswith('<table'):
                        display_container.markdown(output, unsafe_allow_html=True)
                    # Otherwise, display as text
                    else:
                        display_container.write(output)

            # Display images
            if 'image_urls' in part and part['image_urls']:
                for url in part['image_urls']:
                    try:
                        response = requests.get(url)
                        img = Image.open(BytesIO(response.content))
                        display_container.image(img, use_column_width=True)
                    except Exception as e:
                        display_container.error(f"Error displaying image from {url}: {e}")

    elif processed_response['type'] == 'processed':
        # Handle processed response with tables and images
        # First display the text in the current context (left column)
        # Only display text in the left column (st), not in the visualization container
        if container:
            # Don't display text in the visualization container
            pass
        else:
            st.markdown(processed_response['text'])

        # Display tables in the visualization container
        if processed_response['tables']:
            print(f"--- Debug: Displaying {len(processed_response['tables'])} tables in visualization container ---")
            for i, df in enumerate(processed_response['tables']):
                with display_container.expander(f"Table {i+1}", expanded=True):
                    display_container.dataframe(df)

                    # Add download button for each table with a unique key
                    csv = df.to_csv(index=False).encode('utf-8')
                    import uuid
                    unique_key = f"download_table_processed_{i+1}_{uuid.uuid4()}"
                    display_container.download_button(
                        label=f"Download Table {i+1} as CSV",
                        data=csv,
                        file_name=f"table_{i+1}.csv",
                        mime="text/csv",
                        key=unique_key
                    )
        else:
            print(f"--- Debug: No tables found to display in visualization container ---")

        # Display generated code if available
        if 'generated_code_file' in processed_response and processed_response['generated_code_file']:
            display_generated_code(processed_response, display_container)

        # Display HTML tables in the visualization container
        if processed_response['html_tables']:
            for i, html in enumerate(processed_response['html_tables']):
                with display_container.expander(f"HTML Table {i+1}", expanded=True):
                    display_container.markdown(html, unsafe_allow_html=True)

        # Display images in the visualization container
        if processed_response['image_urls']:
            for url in processed_response['image_urls']:
                try:
                    response = requests.get(url)
                    img = Image.open(BytesIO(response.content))
                    display_container.image(img, use_column_width=True)
                except Exception as e:
                    display_container.error(f"Error displaying image from {url}: {e}")

    else:
        # Handle plain text with tables and images
        # First display the text in the current context (left column)
        # Only display text in the left column (st), not in the visualization container
        if container:
            # Don't display text in the visualization container
            pass
        else:
            st.markdown(processed_response['text'])

        # Then display tables in the visualization container (right column)
        if processed_response['tables']:
            print(f"--- Debug: Displaying {len(processed_response['tables'])} tables in visualization container ---")
            for i, df in enumerate(processed_response['tables']):
                with display_container.expander(f"Table {i+1}", expanded=True):
                    display_container.dataframe(df)

                    # Add download button for each table with a unique key
                    csv = df.to_csv(index=False).encode('utf-8')
                    import uuid
                    unique_key = f"download_table_plain_{i+1}_{uuid.uuid4()}"
                    display_container.download_button(
                        label=f"Download Table {i+1} as CSV",
                        data=csv,
                        file_name=f"table_{i+1}.csv",
                        mime="text/csv",
                        key=unique_key
                    )
        else:
            print(f"--- Debug: No tables found to display in visualization container ---")

        # Display HTML tables in the visualization container
        if processed_response['html_tables']:
            for i, html in enumerate(processed_response['html_tables']):
                with display_container.expander(f"HTML Table {i+1}", expanded=True):
                    display_container.markdown(html, unsafe_allow_html=True)

        # Display images in the visualization container
        if processed_response['image_urls']:
            for url in processed_response['image_urls']:
                try:
                    response = requests.get(url)
                    img = Image.open(BytesIO(response.content))
                    display_container.image(img, use_column_width=True)
                except Exception as e:
                    display_container.error(f"Error displaying image from {url}: {e}")

def display_generated_code_files(container=None):
    """
    Display all generated code files from the outputs directory.
    
    Args:
        container: Container to display in (optional)
    """
    # Use the provided container or the current context
    display_container = container if container else st

    # Check if outputs directory exists
    if not os.path.exists("outputs"):
        return

    # Get all generated code files
    code_files = [f for f in os.listdir("outputs") if f.startswith("generated_code")]

    if code_files:
        display_container.markdown("### Generated Code")
        
        for file in code_files:
            file_path = os.path.join("outputs", file)
            
            try:
                # Read the code file
                with open(file_path, "r") as f:
                    code_content = f.read()
                
                # Try to extract code from JSON format
                try:
                    code_json = json.loads(code_content)
                    if "python" in code_json:
                        code_content = code_json["python"]
                except:
                    # If not JSON, use as is
                    pass
                
                # Display the code with syntax highlighting
                with display_container.expander(f"Code: {file}", expanded=True):
                    display_container.code(code_content, language="python")
                    
                    # Add a copy button
                    if display_container.button(f"Copy Code", key=f"copy_{file}"):
                        # Use JavaScript to copy to clipboard
                        import uuid
                        temp_id = f"code_{uuid.uuid4()}"
                        display_container.markdown(
                            f"""
                            <textarea id="{temp_id}" style="position: absolute; left: -9999px;">{code_content}</textarea>
                            <script>
                                var copyText = document.getElementById("{temp_id}");
                                copyText.select();
                                document.execCommand("copy");
                            </script>
                            """,
                            unsafe_allow_html=True
                        )
                        display_container.success("Code copied to clipboard!")
            except Exception as e:
                display_container.error(f"Error displaying code file {file}: {e}")

def display_output_images(container=None):
    """
    Display all images from the outputs directory.
    
    Args:
        container: Container to display in (optional)
    """
    # Use the provided container or the current context
    display_container = container if container else st

    # Check if outputs directory exists
    if not os.path.exists("outputs"):
        return

    # Get all image files
    image_files = [f for f in os.listdir("outputs") if f.lower().endswith((".png", ".jpg", ".jpeg", ".gif"))]

    if image_files:
        display_container.markdown("### Generated Images")
        
        for img_file in image_files:
            file_path = os.path.join("outputs", img_file)
            
            try:
                # Display the image
                display_container.image(file_path, caption=img_file, use_column_width=True)
                
                # Add a download button for the image
                with open(file_path, "rb") as f:
                    img_bytes = f.read()
                    import uuid
                    display_container.download_button(
                        label=f"Download {img_file}",
                        data=img_bytes,
                        file_name=img_file,
                        mime=f"image/{img_file.split('.')[-1]}",
                        key=f"download_{uuid.uuid4()}"
                    )
            except Exception as e:
                display_container.error(f"Error displaying image {img_file}: {e}")