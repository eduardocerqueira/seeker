#date: 2024-04-26T17:07:35Z
#url: https://api.github.com/gists/31cf8b58ca850312f883ab86fbaca361
#owner: https://api.github.com/users/rakinplaban

import os
import re
import pandas as pd
from pprint import pprint
import io


# Define data cleaning function
def clean_data(df):
    # Example: Remove leading and trailing whitespaces from all columns
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    # Example: Drop rows with missing values
    df.dropna(inplace=True)
    return df

# Define function to clean CSV files in a directory
def clean_csv_files_in_directory(input_directory, output_directory):
    for root, dirs, files in os.walk(input_directory):
        for file in files:
            if file.endswith('.csv'):
                input_file_path = os.path.join(root, file)
                output_file_path = os.path.join(output_directory, os.path.relpath(input_file_path, input_directory))
                output_file_dir = os.path.dirname(output_file_path)
                os.makedirs(output_file_dir, exist_ok=True)
                print("Cleaning:", input_file_path)
                # Read CSV file line by line and skip problematic rows
                with open(input_file_path, 'r', encoding='utf-8') as infile:
                    lines = infile.readlines()
                    cleaned_lines = []
                    for line in lines:
                        try:
                            df = pd.read_csv(io.StringIO(line), header=None)
                            cleaned_lines.append(line)
                        except pd.errors.ParserError:
                            print("Skipped line due to parsing error:", line)
                    # Write cleaned lines to a new CSV file
                    with open(output_file_path, 'w', encoding='utf-8') as outfile:
                        outfile.writelines(cleaned_lines)
                print("Cleaning completed. Saved to:", output_file_path)

# Define the input directory containing all data
input_root_directory = 'path_to_unclear_csv'

# Define the output directory for cleaned data
output_root_directory = 'path_to_clear_csv_will_be_saved'

# Clean CSV files in the input directory and its subdirectories, and save to the output directory
clean_csv_files_in_directory(input_root_directory, output_root_directory)
#
print("All CSV files cleaned successfully and saved to the output directory.")