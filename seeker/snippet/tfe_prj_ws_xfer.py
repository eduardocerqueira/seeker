#date: 2024-02-14T16:51:01Z
#url: https://api.github.com/gists/1a358cc3f2e78eab0f4f0e843934a563
#owner: https://api.github.com/users/bry5an

import requests
import argparse
import os

# Set the base URL for the Terraform Enterprise API
base_url = 'https://app.terraform.io/api/v2'

# Set up argument parser
parser = argparse.ArgumentParser(description='Move all workspaces from a given project within Terraform Enterprise to another project.')
parser.add_argument('-o', '--organization', help='The name of the organization')
parser.add_argument('-s', '--source_project', help='The name of the source project')
parser.add_argument('-d', '--destination_project', help='The name of the destination project')
args = parser.parse_args()

 "**********"i "**********"f "**********"  "**********"' "**********"T "**********"F "**********"E "**********"_ "**********"T "**********"O "**********"K "**********"E "**********"N "**********"' "**********"  "**********"n "**********"o "**********"t "**********"  "**********"i "**********"n "**********"  "**********"o "**********"s "**********". "**********"e "**********"n "**********"v "**********"i "**********"r "**********"o "**********"n "**********": "**********"
    print("Error: "**********"
    exit(1)

# Set the headers for the API requests
headers = {
    'Authorization': "**********"
    'Content-Type': 'application/vnd.api+json',
}

# Get the workspaces in the source project
response = requests.get(f'{base_url}/organizations/{args.organization}/workspaces?filter[workspace][name]={args.source_project}', headers=headers)
source_workspaces = response.json()['data']

# For each workspace in the source project, move it to the destination project
for workspace in source_workspaces:
    workspace_id = workspace['id']
    payload = {
        'data': {
            'type': 'workspaces',
            'attributes': {
                'name': workspace['attributes']['name'],
                'organization': {
                    'name': args.organization
                },
                'allow-destroy-plan': workspace['attributes']['allow-destroy-plan']
            },
            'relationships': {
                'terraform-version': {
                    'data': {
                        'type': 'terraform-versions',
                        'id': workspace['relationships']['terraform-version']['data']['id']
                    }
                },
                'vcs-repo': {
                    'data': {
                        'identifier': args.destination_project,
                        'oauth-token-id': "**********"
                        'branch': workspace['attributes']['vcs-repo']['branch'],
                        'default-branch': workspace['attributes']['vcs-repo']['default-branch'],
                        'ingress-submodules': workspace['attributes']['vcs-repo']['ingress-submodules']
                    }
                }
            }
        }
    }
    response = requests.patch(f'{base_url}/workspaces/{workspace_id}', headers=headers, json=payload)
    if response.status_code != 200:
        print(f'Failed to move workspace {workspace_id} to project {args.destination_project}')