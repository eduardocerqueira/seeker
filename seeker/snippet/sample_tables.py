#date: 2023-04-12T17:08:11Z
#url: https://api.github.com/gists/b6e3baebadce423f0f1a440f5e262ff8
#owner: https://api.github.com/users/zackbunch

import requests
import json
import pandas as pd

# Set the Confluence REST API URL and credentials
api_url = "https://your-confluence-url.com/rest/api/content/"
auth = "**********"

# Load the DataFrame from a file or create it
df = pd.read_csv("example.csv")

# Group the DataFrame by department
groups = df.groupby("Department")

# Create a new Confluence page for each department
for department, group_df in groups:
    # Define the page title and parent page ID
    title = department
    parent_id = 12345  # Replace with the ID of the parent page
    
    # Define the page body as the Confluence markup table
    markup = "|"
    markup += " | ".join(group_df.columns) + " |\n"
    for _, row in group_df.iterrows():
        markup += "| " + " | ".join(str(val) for val in row.values) + " |\n"
    
    # Define the Confluence page content as a JSON object
    page_data = {
        "title": title,
        "type": "page",
        "ancestors": [{"id": parent_id}],
        "body": {"storage": {"value": markup, "representation": "storage"}}
    }
    
    # Convert the page data to JSON and send the POST request to create the page
    headers = {"Content-Type": "application/json"}
    response = requests.post(api_url, auth=auth, headers=headers, data=json.dumps(page_data))
    
    # Check the response status code and print a message
    if response.status_code == 200:
        print(f"Created page '{title}' with ID {json.loads(response.text)['id']}.")
    else:
        print(f"Error creating page '{title}'. Status code: {response.status_code}")
sponse.status_code}")
