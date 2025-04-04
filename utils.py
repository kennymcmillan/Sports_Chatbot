import json
import pandas as pd
import re
import os

def process_julius_response(response_text):
    """
    Process the response from Julius AI to extract structured data like tables and images.
    Returns a dictionary with the parsed components.
    """
    print(f"--- Debug: Processing response (length: {len(response_text)}) ---")
    
    # Extract pure text by removing all JSON-like patterns
    pure_text = response_text
    
    # First, try to find and extract the JSON structure with outputs
    outputs = []
    tables = []
    html_tables = []
    image_urls = []
    generated_code_file = None # To store path to generated code file

    # Check for template variable pattern like {{outputs[0]}}
    template_pattern = r'\{\{outputs\[(\d+)\]\}\}'
    template_matches = re.findall(template_pattern, response_text)
    if template_matches:
        print(f"--- Debug: Found template variable pattern {{{{outputs[{template_matches[0]}]}}}} ---")
        # This indicates there should be tabular data, but it's not properly formatted
        # Let's try to create a simple dataframe from the text content
        try:
            # Extract the text that might contain the table data
            lines = response_text.split('\n')
            table_start_idx = -1
            for i, line in enumerate(lines):
                if '{{outputs[' in line:
                    table_start_idx = i
                    break
            
            if table_start_idx >= 0:
                # Look for table-like content after the template variable
                table_lines = []
                for i in range(table_start_idx + 1, len(lines)):
                    if lines[i].strip() and not lines[i].strip().startswith("Let me know"):
                        table_lines.append(lines[i])
                    elif len(table_lines) > 0 and lines[i].strip().startswith("Let me know"):
                        break
                
                if table_lines:
                    # Try to parse as CSV-like content
                    import io
                    import pandas as pd
                    table_text = '\n'.join(table_lines)
                    try:
                        # Try comma separator first
                        df = pd.read_csv(io.StringIO(table_text))
                        tables.append(df)
                        print(f"--- Debug: Successfully created dataframe from text content ---")
                    except:
                        try:
                            # Try tab separator
                            df = pd.read_csv(io.StringIO(table_text), sep='\t')
                            tables.append(df)
                            print(f"--- Debug: Successfully created dataframe from text content (tab-separated) ---")
                        except Exception as e:
                            print(f"--- Debug: Failed to create dataframe from text: {e} ---")
        except Exception as e:
            print(f"--- Debug: Error processing template variable: {e} ---")
    
    # Look for the {"outputs": [...]} pattern which is the most common format
    outputs_pattern = r'\{"outputs":\s*\[(.*?)\]\}'
    outputs_matches = re.findall(outputs_pattern, response_text, re.DOTALL)
    
    # Also look for the {"output": [...]} pattern (note: output vs outputs)
    output_pattern = r'\{"output":\s*\[(.*?)\]\}'
    output_matches = re.findall(output_pattern, response_text, re.DOTALL)
    
    if output_matches:
        print(f"--- Debug: Found {len(output_matches)} output patterns ---")
        # Add output matches to outputs_matches for processing
        outputs_matches.extend(output_matches)
    
    if outputs_matches:
        print(f"--- Debug: Processing {len(outputs_matches)} total output patterns ---")
        
        for match in outputs_matches:
            # Clean up the match to make it valid JSON
            cleaned_match = match.strip()
            if not cleaned_match.endswith(']'):
                cleaned_match += ']'
            
            try:
                # Try to parse each output item individually
                items = []
                # Split by commas but respect quotes and brackets
                in_quotes = False
                in_brackets = 0
                current_item = ""
                
                for char in cleaned_match:
                    if char == '"' and (len(current_item) == 0 or current_item[-1] != '\\'):
                        in_quotes = not in_quotes
                    elif char == '[':
                        in_brackets += 1
                    elif char == ']':
                        in_brackets -= 1
                    
                    if char == ',' and not in_quotes and in_brackets == 0:
                        items.append(current_item.strip())
                        current_item = ""
                    else:
                        current_item += char
                
                if current_item:
                    items.append(current_item.strip())
                
                # Process each item
                for item in items:
                    print(f"--- Debug: Processing item: {item[:50]}... ---")
                    
                    # Special case for items that start with a quote and then [JULIUS_TABLE]
                    if item.startswith('"[JULIUS_TABLE]'):
                        print(f"--- Debug: Found item starting with quote and [JULIUS_TABLE] ---")
                        # Remove the outer quotes
                        if item.startswith('"') and item.endswith('"'):
                            item = item[1:-1]
                        
                    # Check if it's a table
                    if '[JULIUS_TABLE]' in item:
                        print(f"--- Debug: Found [JULIUS_TABLE] in item ---")
                        table_pattern = r'\[JULIUS_TABLE\]:\s*"(.*?)"'
                        table_match = re.search(table_pattern, item, re.DOTALL)
                        if table_match:
                            try:
                                # Get the table JSON string and unescape it
                                table_json = table_match.group(1)
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
                                tables.append(df)
                                
                                # Remove the table from the text
                                pure_text = pure_text.replace(item, f"[TABLE {len(tables)}]")
                            except Exception as e:
                                print(f"Error processing table: {e}")
                        
                        # Check if it's an HTML table
                        elif item.startswith('"<table') and item.endswith('table>"'):
                            try:
                                # Clean up the HTML table string
                                html_table = item.strip('"').replace('\\"', '"').replace('\\n', '\n')
                                html_tables.append(html_table)
                                
                                # Remove the HTML table from the text
                                pure_text = pure_text.replace(item, f"[HTML_TABLE {len(html_tables)}]")
                            except Exception as e:
                                print(f"Error processing HTML table: {e}")
                        
                        # Add to outputs if it's not already processed
                        outputs.append(item)
                
            except Exception as e:
                print(f"--- Debug: Error processing outputs match: {e} ---")
    
    # Look for image URLs
    image_url_pattern = r'https?://[^\s"\']+\.(jpg|jpeg|png|gif)'
    image_url_matches = re.findall(image_url_pattern, response_text)
    if image_url_matches:
        for url in image_url_matches:
            image_urls.append(url)
            
    # Remove the JSON structure from the text to get pure text
    pure_text = re.sub(r'\{"outputs":\s*\[.*?\]\}', '', pure_text, flags=re.DOTALL)
    
    # Also look for direct table tags in the text (outside of JSON structure)
    table_pattern = r'\[JULIUS_TABLE\]: "(.*?)"'
    direct_table_matches = re.findall(table_pattern, response_text)
    print(f"--- Debug: Found {len(direct_table_matches)} direct table matches ---")
    
    # Process each direct table
    for i, table_json in enumerate(direct_table_matches):
        try:
            # Unescape the JSON string - multiple levels may be needed
            while '\\\\' in table_json:
                table_json = table_json.replace('\\\\', '\\')
            table_json = table_json.replace('\\"', '"')
            
            print(f"--- Debug: Unescaped direct table JSON (first 100 chars): {table_json[:100]} ---")
            
            # Parse the JSON with error handling
            try:
                table_data = json.loads(table_json)
            except json.JSONDecodeError as e:
                print(f"--- Debug: JSON decode error in direct table: {e} ---")
                # Try to find a valid JSON substring
                if "Extra data" in str(e):
                    # Find the position of the error
                    error_pos = int(str(e).split("column ")[1].split(" ")[0])
                    print(f"--- Debug: Error position: {error_pos} ---")
                    # Try parsing just up to the error position
                    try:
                        table_data = json.loads(table_json[:error_pos])
                    except:
                        # Try to find the last valid JSON object
                        last_brace = table_json.rfind('}')
                        if last_brace > 0:
                            try:
                                table_data = json.loads(table_json[:last_brace+1])
                            except Exception as e2:
                                print(f"--- Debug: Failed to parse JSON: {e2} ---")
                                raise e
                        else:
                            raise e
                else:
                    raise e
            
            # Create a DataFrame
            df = pd.DataFrame(
                data=table_data['data'],
                columns=table_data.get('columns', []),
                index=table_data.get('index', None)
            )
            
            tables.append(df)
            print(f"--- Debug: Successfully created DataFrame from direct table match with shape {df.shape} ---")
            
            # Replace the table tag with a placeholder
            table_placeholder = f'[JULIUS_TABLE]: "{table_json}"'
            pure_text = pure_text.replace(table_placeholder, f"[TABLE {len(tables)}]")
        except Exception as e:
            print(f"Error processing direct table: {e}")
    
    # Check if there's a file at outputs/generated_code_1.txt
    if os.path.exists("outputs/generated_code_1.txt") and not generated_code_file:
        generated_code_file = "outputs/generated_code_1.txt"
        print(f"--- Debug: Found generated code file at {generated_code_file} ---")

    # Return processed response
    return {
        'type': 'processed',
        'text': pure_text.strip(),
        'outputs': outputs,
        'tables': tables,
        'html_tables': html_tables,
        'image_urls': image_urls,
        'raw_response': response_text, # Store raw response
        'generated_code_file': generated_code_file # Path to generated code file
    }