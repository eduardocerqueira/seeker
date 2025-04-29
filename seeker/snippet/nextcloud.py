#date: 2025-04-29T17:06:52Z
#url: https://api.github.com/gists/1ce7d231e1793ed770fdfba9b44a50a3
#owner: https://api.github.com/users/dolohow

import sys

import requests
from requests.auth import HTTPBasicAuth

# Configuration
NEXTCLOUD_URL = ''
USERNAME = ''
PASSWORD = "**********"
TARGET_FOLDER = '/'

API_ENDPOINT = f'{NEXTCLOUD_URL}/ocs/v2.php/apps/files_sharing/api/v1/shares'

# Headers for OCS API
HEADERS = {
    'OCS-APIRequest': 'true',
    'Accept': 'application/json'
}

def get_shares():
    """Fetch all shares from Nextcloud."""
    response = "**********"=HTTPBasicAuth(USERNAME, PASSWORD), headers=HEADERS)
    if response.status_code != 200:
        print(f"Failed to fetch shares: {response.status_code}, {response.text}")
        return []
    
    shares = response.json().get('ocs', {}).get('data', [])
    return shares

def remove_share(share_id):
    """Remove a specific share by its ID."""
    response = "**********"=HTTPBasicAuth(USERNAME, PASSWORD), headers=HEADERS)
    if response.status_code == 200:
        print(f"Successfully removed share ID: {share_id}")
    else:
        print(f"Failed to remove share ID {share_id}: {response.status_code}, {response.text}")

def main():
    shares = get_shares()
    if not shares:
        print("No shares found.")
        return
    
    for share in shares:
        path = share.get('path', '')
        share_id = share.get('id', '')
        
        if path.startswith(TARGET_FOLDER):
            print(f"Found share for target folder: {path} (ID: {share_id})")
            remove_share(share_id)
    
    print("Completed removing shares for the target folder.")

if __name__ == "__main__":
    main()
ed removing shares for the target folder.")

if __name__ == "__main__":
    main()
