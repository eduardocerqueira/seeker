#date: 2026-02-26T17:44:33Z
#url: https://api.github.com/gists/015f6c66571e5fb3dd665494d7fc080d
#owner: https://api.github.com/users/shavera

"""
   Copyright 2026 Alexander Shaver

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

"""
This script downloads all the files from a remarkable tablet that is connected to the computer via the USB
storage device and the usual web interface.

It downloads the files (by default, though feel free to change, of course) to ~/Documents/RemarkableBackup/YYYY-MM-DD/

This was mostly just reverse-engineered from the web interface, so no guarantees it works for all cases.
"""

from datetime import datetime
from pathlib import Path

import requests

BASE_URL = "http://10.11.99.1/documents"

def download_file(url, filepath):
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(filepath, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        print(f"File downloaded successfully: {filepath}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e} - for file {filepath}")

OUTPUT_DIR = Path.home() / "Documents" / "ReMarkableBackup" / datetime.now().strftime("%Y-%m-%d")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def fetch_and_handle(collection_id: str | None = None, dir_path: Path = OUTPUT_DIR):
    url = f"{BASE_URL}/" if collection_id is None else f"{BASE_URL}/{collection_id}"
    response = requests.get(url)
    handle_collection_response(response, dir_path)

def sanitize_name(name: str) -> str:
    return name.replace("/", "_").replace(":", "_").replace(" ", "_")

def handle_collection_response(response: requests.Response, dir_path: Path):
    for doc in response.json():
        doc_type = doc["Type"]
        match doc_type:
            case "DocumentType":
                handle_document(doc, dir_path)
            case "CollectionType":
                san_name = sanitize_name(doc["VissibleName"])
                msg_sub = "" if san_name == doc["VissibleName"] else f"(sanitized: {san_name})"
                print(f"collection: {doc['VissibleName']} {msg_sub}")
                subdir_path = dir_path / san_name
                subdir_path.mkdir(parents=True, exist_ok=True)
                fetch_and_handle(doc["ID"], subdir_path)
            case _:
                print(f"unknown: {doc}")


def handle_document(doc, download_dir: Path):
    print(f"doc: {doc["VissibleName"]} id: {doc['ID']}")
    name = sanitize_name(doc["VissibleName"])
    if doc["fileType"] not in ["notebook", "pdf"]:
        print(f"Unknown file type: {doc['fileType']}")
    ext = "rmdoc" if doc["fileType"] == "notebook" else "pdf"
    download_file(f"{BASE_URL}/download/{doc['ID']}/{ext}", download_dir / f"{name}.{ext}")


if __name__ == "__main__":
    fetch_and_handle()