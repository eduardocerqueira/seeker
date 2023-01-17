#date: 2023-01-17T17:05:16Z
#url: https://api.github.com/gists/0119bd2abb528f7f0b1c27e38a2aa72a
#owner: https://api.github.com/users/woodenzen

import os
import re

def search_incoming(UUID, folder_path):
    files_with_uuid = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if os.path.splitext(file)[1] == ".md":
                with open(os.path.join(root, file), 'r') as f: 
                    data=f.read()
                    if re.search(UUID, data):
                        files_with_uuid.append(file)
    return files_with_uuid

if __name__ == "__main__":
    print(search_incoming('201910281718', '/Users/will/Dropbox/zettelkasten'))