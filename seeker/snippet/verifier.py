#date: 2024-04-24T16:50:27Z
#url: https://api.github.com/gists/e15f924f38159f28e358081f9ad1d225
#owner: https://api.github.com/users/MisterDaniels

import csv
import requests
import sys

url = 'https://api.dropboxapi.com/2/files/list_folder'
headers = {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer ' + sys.argv[1]
}
errors = []

with open(sys.argv[2], newline='') as csv_file:
    csv_reader = csv.reader(csv_file)
    for line in csv_reader:
        if line[0] == 'path':
            continue

        dropbox_path = line[0]
        data = {
            "include_deleted": True,
            "include_has_explicit_shared_members": False,
            "include_media_info": True,
            "include_mounted_folders": True,
            "include_non_downloadable_files": True,
            "path": dropbox_path,
            "recursive": False
        }

        res = requests.post(url, headers=headers, json=data)
        json = res.json()
        
        if 'error_summary' not in json:
            errors.append({
                'path': dropbox_path
            })
            continue

        if 'error_summary' in json:
            if 'path/not_folder' not in json['error_summary']:
                errors.append({
                    'path': dropbox_path
                })
                
if errors:
    print('Images not in folder:')
    for error in errors:
        print(f"File: {error['path']}")
else:
    print('All images in folder!')