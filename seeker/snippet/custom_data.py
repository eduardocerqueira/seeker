#date: 2025-02-11T17:02:09Z
#url: https://api.github.com/gists/aa582b938e1c3391f18c65af22ba79d0
#owner: https://api.github.com/users/Sdy603

import pandas as pd
import requests
import json

# Load the CSV file
file_path = 'df_getrunresults.csv'
df = pd.read_csv(file_path)

# Define the Custom Data API endpoint and API key
API_ENDPOINT = 'https://yourinstance.getdx.net/api/customData.setAll'  # Replace with the actual endpoint
API_KEY = "**********"

# Headers for the API request
headers = {
    'Accepts': 'application/json',
    'Authorization': f'Bearer {API_KEY}',
    'Content-Type': 'application/json'
}

# Function to format the data as required by the Custom Data API
def format_payload(row):
    return {
        "reference": f"{row['id']}",  # Using id as part of the reference
        "key": f"status-{row['test_id']}",  # Using test_id as the key
        "value": {
            "status_id": row['status_id'],
            "created_on": row['created_on'],
            "assignedto_id": row['assignedto_id'] if pd.notna(row['assignedto_id']) else None,
            "comment": row['comment'] if pd.notna(row['comment']) else "",
            "version": row['version'] if pd.notna(row['version']) else "N/A",
            "elapsed": row['elapsed'] if pd.notna(row['elapsed']) else "0",
            "defects": row['defects'] if pd.notna(row['defects']) else "None",
            "created_by": row['created_by'],
            "custom_step_results": row['custom_step_results'] if pd.notna(row['custom_step_results']) else None,
            "custom_chaos_testing_comments": row['custom_chaos_testing_comments'] if pd.notna(row['custom_chaos_testing_comments']) else None,
            "attachment_ids": json.loads(row['attachment_ids']) if pd.notna(row['attachment_ids']) else []
        },
        "timestamp": row['created_on']
    }

# Create the payload with all data
payload = {
    "data": [format_payload(row) for index, row in df.iterrows()]
}

# Send the data to the API
response = requests.post(API_ENDPOINT, headers=headers, json=payload)

# Log the response
if response.status_code == 200:
    print('Successfully sent data to the Custom Data API')
else:
    print(f'Failed to send data: {response.status_code}, {response.text}')
data: {response.status_code}, {response.text}')
