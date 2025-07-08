#date: 2025-07-08T17:11:13Z
#url: https://api.github.com/gists/e6efa459b18fe730d437a07b6b218092
#owner: https://api.github.com/users/Hydramus

import os
import datetime
import argparse
import pandas as pd
from pdpyras import APISession, PDClientError
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.worksheet.table import Table, TableStyleInfo
from openpyxl import load_workbook
from openpyxl.pivot.table import Table as PivotTable

# Set up the argument parser
parser = argparse.ArgumentParser(
    description="PagerDuty Incident Report Script",
    epilog="Ensure you have set the PD_API_KEY environment variable before running the script."
)
parser.add_argument(
    "--email",
    required=True,
    help="The email address to use as the default sender."
)

# Parse the arguments
args = parser.parse_args()

# Retrieve the API key from environment variables
api_key = os.environ.get('PD_API_KEY')
if not api_key:
    parser.error("PD_API_KEY environment variable not set. "
                 "Please set it by running 'export PD_API_KEY=\"your_api_key_here\"'.")

# Use the email argument provided in the command line
default_email = args.email

# Ask the user if they want verbose output
verbose = input("Do you want verbose output? (yes/no): ").lower() == "yes"

# Create the API session with the provided email
session = APISession(api_key, default_from=default_email)

# Define the start and end dates for the time range
start_date = datetime.datetime(2024, 1, 1)
end_date = datetime.datetime.now()

# Calculate the number of days between the start and end dates
num_days = (end_date - start_date).days

# Get the current date for the filename
current_date = datetime.datetime.now().strftime("%Y-%m-%d")
filename = f'pagerduty-incidents-report_{current_date}.xlsx'

# Create a list to hold all incident data
incident_data = []

# Loop over each day in the time range
for i in range(num_days):
    # Calculate the 'since' and 'until' parameters for this day
    since = (start_date + datetime.timedelta(days=i)).isoformat()
    until = (start_date + datetime.timedelta(days=i + 1)).isoformat()

    # Retrieve incidents for this day
    try:
        for incident in session.iter_all('incidents', params={'since': since, 'until': until}):
            # Calculate the duration of the incident
            created_at = datetime.datetime.strptime(incident['created_at'], "%Y-%m-%dT%H:%M:%SZ")
            resolved_at = datetime.datetime.strptime(incident['last_status_change_at'], "%Y-%m-%dT%H:%M:%SZ") if incident['status'] == 'resolved' else datetime.datetime.now()
            duration = (resolved_at - created_at).total_seconds() / 3600  # Convert duration to hours

            # Append incident details to the list
            incident_data.append({
                'ID': incident['id'],
                'Status': incident['status'],
                'Priority': incident['priority']['name'] if incident['priority'] and incident['priority'].get('name') else '',
                'Urgency': incident['urgency'],
                'Title': incident['title'],
                'Created': incident['created_at'],
                'Service': incident['service']['name'] if incident['service'] and incident['service'].get('name') else '',
                'Assigned To': ', '.join([assignment['assignee']['name'] for assignment in incident['assignments'] if assignment['assignee'] and assignment['assignee'].get('name')]),
                'URL': f"https://app.pagerduty.com/incidents/{incident['id']}",
                'Duration (hours)': duration
            })

            # If verbose output is enabled, print the incident details
            if verbose:
                print(f"Exported incident {incident['id']} with status '{incident['status']}', priority '{incident['priority']['name'] if incident['priority'] and incident['priority'].get('name') else ''}', urgency '{incident['urgency']}', title '{incident['title']}', created at '{incident['created_at']}', service '{incident['service']['name'] if incident['service'] and incident['service'].get('name') else ''}', assigned to '{', '.join([assignment['assignee']['name'] for assignment in incident['assignments'] if assignment['assignee'] and assignment['assignee'].get('name')])}', URL: https://app.pagerduty.com/incidents/{incident['id']}, duration: {str(duration)} hours")
    except PDClientError as e:
        print(f"Failed to retrieve incidents for {since} to {until}: {e}")

# Convert the list of incident data to a DataFrame
df = pd.DataFrame(incident_data)

# Write the DataFrame to an Excel file
with pd.ExcelWriter(filename, engine='openpyxl') as writer:
    df.to_excel(writer, sheet_name='Incidents', index=False)

    # Load the workbook to add more features
    workbook = writer.book
    worksheet = writer.sheets['Incidents']

    # Create a Table style for better formatting
    table = Table(displayName="IncidentTable", ref=f"A1:{chr(64 + len(df.columns))}{len(df) + 1}")
    style = TableStyleInfo(name="TableStyleMedium9", showFirstColumn=False, showLastColumn=False, showRowStripes=True, showColumnStripes=True)
    table.tableStyleInfo = style
    worksheet.add_table(table)

    # Create a Pivot Table
    pivot_ws = workbook.create_sheet(title="PivotTable")
    worksheet_data = workbook['Incidents']
    pivot = PivotTable(name="PivotTable1", anchor="A1", ref="A1:K" + str(len(df) + 1))
    pivot.add_field(name="Service")
    pivot.add_field(name="Status")
    pivot_ws.add_table(pivot)

print(f"Excel file '{filename}' with incidents report and pivot table created successfully.")
