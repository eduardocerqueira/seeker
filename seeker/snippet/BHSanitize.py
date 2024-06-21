#date: 2024-06-21T16:47:02Z
#url: https://api.github.com/gists/b827745e8757f2ddce78934e568e7ae9
#owner: https://api.github.com/users/watson0x90

# Script Name: BHSanitize.py
# Author: Ryan Watson
# Gist Github: https://gist.github.com/Watson0x90
# Created on: 2024-06-21
# Last Modified: 024-06-21
# Description: Designed to sanitize BloodHound JSON data by replacing Unicode escape sequences `\u` with `uni-` and base64 encoding invalid characters.
# Version: 1.0.0
# License: MIT License
# Usage: python BHSanitize.py

import os
import json
import re
import base64

# Function to replace Unicode escape sequences
def replace_unicode_escapes(text):
    return re.sub(r'\\u([0-9A-Fa-f]{4})', r'uni-\1', text, flags=re.IGNORECASE)

# Function to base64 encode a string
def base64_encode(value):
    return base64.b64encode(value.encode('utf-8')).decode('utf-8')

# Function to check for invalid characters and encode if necessary
def check_and_encode_json(json_data):
    if isinstance(json_data, dict):
        return {k: check_and_encode_json(v) for k, v in json_data.items()}
    elif isinstance(json_data, list):
        return [check_and_encode_json(item) for item in json_data]
    elif isinstance(json_data, str):
        if '�' in json_data:
            return base64_encode(json_data)
    return json_data

# Function to process JSON files in a folder
def process_json_files(folder_path, debug=False):
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            
            # Read the JSON file
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                
                # Replace Unicode escape sequences
                updated_content = replace_unicode_escapes(content)
                
                # Attempt to load JSON data to ensure valid JSON format
                try:
                    json_data = json.loads(updated_content)
                except json.JSONDecodeError as e:
                    if debug:
                        print(f"Failed to decode JSON in file: {file_path}, error: {e}")
                    continue
                
                # Check for invalid characters and encode if necessary
                json_data = check_and_encode_json(json_data)
                
                # Check if content has changed
                if updated_content != content and debug:
                    print(f"Changes made in file: {file_path}")
                
                # Write the updated content back to the file without pretty-printing
                with open(file_path, 'w', encoding='utf-8') as file:
                    json.dump(json_data, file, ensure_ascii=False)
                
                print(f"Processed file: {file_path}")

# Main function
def main():
    folder_path = r"/your/path/here"  # Update with the path to your JSON folder
    debug = True  # Set this to True to enable debug output
    process_json_files(folder_path, debug)
    print("Processing completed.")

if __name__ == "__main__":
    main()
