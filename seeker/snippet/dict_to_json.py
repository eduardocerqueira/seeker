#date: 2025-03-28T16:49:09Z
#url: https://api.github.com/gists/4e3d495d887b9abe948e8a43fac63e39
#owner: https://api.github.com/users/do-me

import json, gzip

def load_json(file_path):
    try:
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            return json.load(f)
    except OSError: #if the file is not gzipped
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

def save_json(data, file_path, compress=True):
    if compress:
        with gzip.open(file_path, 'wt', encoding='utf-8') as f:
            json.dump(data, f)  # Use indent for readability (optional)
    else:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f)
            
save_json(zip_codes, "zip_codes.json.gz", compress=True)

zip_codes = load_json("zip_codes.json.gz")
