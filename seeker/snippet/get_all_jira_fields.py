#date: 2024-09-04T16:38:20Z
#url: https://api.github.com/gists/95237dfa7151d82748aaff7dea48d86b
#owner: https://api.github.com/users/KamikazeQ

import requests
from requests.auth import HTTPBasicAuth
import csv

# Jira API credentials
JIRA_URL = "https://your-jira-instance-url"
USERNAME = "your-jira-username"
PASSWORD = "**********"

# Endpoint to get all fields (including custom fields)
FIELDS_ENDPOINT = f"{JIRA_URL}/rest/api/2/field"

headers = {
    "Accept": "application/json"
}

def fetch_all_fields():
    """
    Fetch all fields (including custom fields) from Jira Data Center.
    :return: List of all fields.
    """
    response = requests.get(
        FIELDS_ENDPOINT,
        headers=headers,
        auth= "**********"
    )
    
    response.raise_for_status()

    fields = response.json()

    print(f"Total fields fetched: {len(fields)}")
    
    return fields

def generate_csv(fields, output_file="fields.csv"):
    """
    Generate a CSV file with field name, type, and custom field ID (without 'customfield_' prefix).
    """
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Column order: Field Name, Field Type, Custom Field ID (without the 'customfield_' prefix)
        writer.writerow(["Field Name", "Field Type", "Custom Field ID"])

        for field in fields:
            field_name = field.get('name', 'N/A')
            field_type = field.get('schema', {}).get('type', 'N/A')
            field_id = field.get('id', 'N/A').replace('customfield_', '')  # Remove the 'customfield_' prefix
            writer.writerow([field_name, field_type, field_id])

def main():
    try:
        fields = fetch_all_fields()
        if fields:
            generate_csv(fields)
            print(f"CSV file generated with {len(fields)} fields.")
        else:
            print("No fields found.")
    except requests.exceptions.HTTPError as err:
        print(f"HTTP Error: {err}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
_ == "__main__":
    main()
