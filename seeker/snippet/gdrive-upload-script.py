#date: 2022-04-11T16:58:24Z
#url: https://api.github.com/gists/aefcd12c82623a4547a200b0fefb75b6
#owner: https://api.github.com/users/TheGarkine

#!/usr/bin/python3
from collections import namedtuple
from typing import List, Optional

import argparse
import magic

from googleapiclient.discovery import build, Resource
from oauth2client.service_account import ServiceAccountCredentials
from googleapiclient.http import MediaFileUpload
GDriveFile = namedtuple("GDriveFile", "id name")
GDriveFile.__doc__ = "A simple class that represents a Google Drive file."

def get_gdrive_service(token_location: Optional[str] = "token.json") -> Resource:
    """Uses the given location to build a Google Drive service.

    Args:
        token_location: The location of the json file for the token. Defaults to "token.json".

    Returns: 
        The google drive service.
    """
    credentials = ServiceAccountCredentials.from_json_keyfile_name(token_location)
    return build('drive', 'v3', credentials=credentials)

def find_files_in_folder(service: Resource, folder_id: str) -> List[GDriveFile]:
    """Returns all files in a given folder.

    Args:
        service: The gdrive service to use when searching the files.
        folder_id: The folder id, of which parenting should be queried.

    Returns:
        The list of all files in the given folder.
    """
    results = service.files().list(q=f"'{folder_id}' in parents", fields="files(id, name)").execute()
    return  [GDriveFile(f["id"], f["name"]) for f in results.get('files', [])]

def upsert_file(service: Resource, file_name: str, file_location: str, folder_id: str) -> GDriveFile:
    """Upserts a file.
    Creates a new one if a file of the same name does not exist.
    If it already exists it gets updated.

    Args:
        service: The Google Drive service to use for the operation.
        file_name: The filename in google drive.
        file_location: The location of the local file.
        folder_id: The Google Drive folder id for the file to be put in.

    Returns:
        The created Google Drive File.
    """
    mime = magic.Magic(mime=True).from_file(file_location) 

    file_metadata = {'name': file_name, 'parents' : [folder_id]}
    media = MediaFileUpload(file_location, mimetype=mime)

    files = find_files_in_folder(service=service,folder_id=folder_id)
    exist = [f for f in files if f.name == file_name]

    if exist:
        file_id = exist[0].id
        file = service.files().update(fileId=file_id, media_body=media, fields='id').execute()
        return GDriveFile(file["id"], file_name)
    else:
        file = service.files().create(body=file_metadata, media_body=media,fields='id').execute()
        return GDriveFile(file["id"], file_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", required=True, help="The name of the file in Google Drive.")
    parser.add_argument("-l", "--location", required=True, help="The location of the local file.")
    parser.add_argument("-f", "--folder", required=True, help="The folder id to which it should be uploaded.")
    args = parser.parse_args()

    service = get_gdrive_service()
    items = upsert_file(service=service, file_name=args.name, file_location=args.location, folder_id=args.folder)