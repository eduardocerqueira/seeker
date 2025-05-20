#date: 2025-05-20T17:14:09Z
#url: https://api.github.com/gists/a49f7f7bd6ab8dc0f29442b4203362b4
#owner: https://api.github.com/users/wjkennedy

import requests
import configparser

# Load config
config = configparser.ConfigParser()
config.read('config/config.properties')

BASE_URL = config['jira']['base_url']
EMAIL = config['jira']['email']
API_TOKEN = "**********"

HEADERS = {
    "Accept": "application/json",
    "Content-Type": "application/json"
}

AUTH = "**********"

def get_project_roles(project_key):
    url = f"{BASE_URL}/rest/api/3/project/{project_key}/role"
    response = requests.get(url, headers=HEADERS, auth=AUTH)
    response.raise_for_status()
    return response.json()

def add_user_to_role(project_key, role_url, account_id):
    url = role_url
    payload = {
        "user": [account_id]
    }
    response = requests.post(url, json=payload, headers=HEADERS, auth=AUTH)
    response.raise_for_status()
    print(f"Added user to Admin role for project {project_key}")

def process_projects(project_keys, account_id):
    for project_key in project_keys:
        roles = get_project_roles(project_key)
        admin_role_url = None
        for role_name, url in roles.items():
            if role_name.lower() == 'admin' or role_name.lower() == 'administrators':
                admin_role_url = url
                break

        if not admin_role_url:
            print(f"Admin role not found for project {project_key}")
            continue

        try:
            add_user_to_role(project_key, admin_role_url, account_id)
        except requests.HTTPError as e:
            print(f"Error adding user to project {project_key}: {e.response.text}")

# Example usage
if __name__ == "__main__":
    user_account_id = "5b10a2844c20165700ede21g"  # Replace with target user's accountId
    projects = ["PROJ1", "PROJ2", "PROJ3"]         # Replace with your actual project keys
    process_projects(projects, user_account_id)
ts, user_account_id)
