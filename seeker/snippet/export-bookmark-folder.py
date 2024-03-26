#date: 2024-03-26T17:01:15Z
#url: https://api.github.com/gists/870e5ca56c59cd4095c3e13e1eb33832
#owner: https://api.github.com/users/EdoardoTosin

import json

# Folder name (case-sensitive)
folder_name = "Folder Name Here"

def find_folder(bookmarks):
    for item in bookmarks:
        if item.get('type') == 'text/x-moz-place-container' and item.get('title') == folder_name:
            return item
        elif item.get('children'):
            result = find_folder(item['children'])
            if result:
                return result
    return None

with open('bookmarks.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

folder = find_folder(data['children'])

if folder:
    with open(f'{folder_name.lower()}.json', 'w', encoding='utf-8') as outfile:
        json_data = json.dumps(folder, ensure_ascii=False)
        outfile.write(json_data)
    print(f"{folder_name} folder saved to {folder_name.lower()}.json")
else:
    print(f"{folder_name} folder not found.")