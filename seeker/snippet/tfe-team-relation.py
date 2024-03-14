#date: 2024-03-14T17:03:07Z
#url: https://api.github.com/gists/5e52107bae892f765b252d8374dceae7
#owner: https://api.github.com/users/bry5an

import requests
import argparse
import os
from prettytable import PrettyTable

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--org', default='myorg', help='The name of the organization to query.')
args = parser.parse_args()

base_url = f'https://app.terraform.io/api/v2/organizations/{args.organization}'

org_name = args.org

 "**********"i "**********"f "**********"  "**********"' "**********"T "**********"F "**********"E "**********"_ "**********"T "**********"O "**********"K "**********"E "**********"N "**********"' "**********"  "**********"n "**********"o "**********"t "**********"  "**********"i "**********"n "**********"  "**********"o "**********"s "**********". "**********"e "**********"n "**********"v "**********"i "**********"r "**********"o "**********"n "**********": "**********"
    print("Error: "**********"
    exit(1)

# Set the headers for the API request
headers = {
    'Authorization': "**********"
    'Content-Type': 'application/vnd.api+json'
}

# Initialize the table
table = PrettyTable()
table.field_names = ["Workspace Name", "Team Name", "Project Name"]

# Iterate through each page of workspaces
page_number = 1
while True:
    # Get the workspaces for the current page
    workspaces_response = requests.get(f'{base_url}/organizations/myorg/workspaces?page[number]={page_number}', headers=headers)
    workspaces = workspaces_response.json()['data']

    # If there are no more workspaces, break the loop
    if not workspaces:
        break

    # Iterate through each workspace
    for workspace in workspaces:
        workspace_name = workspace['attributes']['name']

        # Get the teams for the current workspace
        teams_response = requests.get(f'{base_url}/workspaces/{workspace_name}/relationships/teams', headers=headers)
        teams = teams_response.json()['data']

        # Iterate through each team
        for team in teams:
            team_name = team['attributes']['name']

            # Get the projects for the current team
            projects_response = requests.get(f'{base_url}/teams/{team_name}/relationships/projects', headers=headers)
            projects = projects_response.json()['data']

            # Iterate through each project
            for project in projects:
                project_name = project['attributes']['name']

                # Add the data to the table
                table.add_row([workspace_name, team_name, project_name])

    # Go to the next page
    page_number += 1

# Print the table
print(table)