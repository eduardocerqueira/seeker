#date: 2024-02-22T16:56:21Z
#url: https://api.github.com/gists/4d436e9a8208b1c2c25db1931a567723
#owner: https://api.github.com/users/jenyeeiam

import csv
import json

# Define the input and output file names
input_csv_file = 'connection_data.csv'
output_json_file = 'connection_data.json'

# Function to preprocess text according to the specified rules
def preprocess_text(text):
    # Replace internal double quotes with two double quotes
    text = text.replace('"', '""')
    # Escape new lines
    text = text.replace('\n', '\\n')
    return text

# Initialize an empty list to hold the converted data
json_data = []

# Open the CSV file for reading with 'utf-8-sig' encoding to handle BOM
with open(input_csv_file, mode='r', encoding='utf-8-sig') as csv_file:
    # Create a CSV reader object
    csv_reader = csv.DictReader(csv_file)
    
    # Iterate over the CSV rows
    for row in csv_reader:
        # Convert the 'label' field from string to integer
        row['label'] = int(row['label'])
        # Preprocess the 'text' field
        row['text'] = preprocess_text(row['text'])
        # Add the row to the list
        json_data.append({'text': f'"{row["text"]}"', 'label': row['label']})

# Open the JSON file for writing
with open(output_json_file, mode='w', encoding='utf-8') as json_file:
    # Write the data to the JSON file in the specified format, ensuring that
    # non-ASCII characters are output as-is and not as Unicode escape sequences.
    json.dump(json_data, json_file, ensure_ascii=False, indent=2)

print(f"CSV data has been converted to JSON and saved to {output_json_file}")
