#date: 2025-08-05T16:39:37Z
#url: https://api.github.com/gists/4a47b8825518147e41e668fab1d71bf9
#owner: https://api.github.com/users/AudIsCool

from base64 import b64decode, b64encode
from datetime import datetime

from io import BytesIO
from json import loads, dumps
from os import walk, getcwd, mkdir, rename
from os.path import join, getmtime, exists
from gzip import GzipFile, compress

import argparse
import time


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='path to backup file')
parser.add_argument('-r', '--remap_path', help='path to remap videos to')

args = parser.parse_args()

def lazily_merge_objects(a: dict, b: dict) -> dict:
    """
        Merges two dictionaries, with the second dictionary taking precedence over the first
    """
    return {**a, **b}

def lazily_check_if_is_backup(obj: dict) -> bool:
    return "videoStore" in obj

def read_backup(filepath: str) -> dict | None:
    """
        Returns the contents of a backup file as a dictionary.
    """
    if filepath.endswith(".json"):
        with open(filepath, 'r') as f:
            obj = loads(f.read())
            if (lazily_check_if_is_backup(obj)): return obj

    else: 
        with open(filepath, 'rb') as f:
            contents = f.read()
            decoded_data = b64decode(contents)
            compressed = BytesIO(decoded_data)
            decompressed = GzipFile(fileobj=compressed)

            obj = loads(decompressed.read().decode('utf-8'))
            if (lazily_check_if_is_backup(obj)): return obj

    return None


def save_backup(filepath: str, data: dict, type: str = "bak") -> None:
    """
        Saves a backup file to the specified path, in either a JSON or BAK format (both can be loaded by Insights Capture)
    """
    if type == "json":
        with open(filepath, 'w') as f:
            f.write(dumps(data, indent=4))
    else:
        text = dumps(data, indent=4).encode('utf-8')
        compressed_data = compress(text)
        encoded_data = b64encode(compressed_data)
        with open(filepath, 'wb') as f:
            f.write(encoded_data)

def get_filename(file):
    return file.split("\\")[-1]

def get_current_unixtime():
    return int(time.time() * 1000)

def updateVideoStore(inputStore, folderToReplace):
    newInputStore = inputStore
    for items in inputStore: 
        video = inputStore[items]
        video_filename = get_filename(video["result"]["file_path"])

        # Update the encoded `url`
        # Important if the user had recordings in a sub-dir then moved them to the main 
        if "url" in video['result']:
            space_less_file_name = video_filename
            while " " in space_less_file_name:
                space_less_file_name = space_less_file_name.replace(" ", "+")
  
            video["result"]["url"] = "overwolf://media/recordings/Insights+Capture\\\\{}".format(video_filename) 

        # Update the `file_path`
        video["result"]["file_path"] = "{}\\{}".format(folderToReplace, video_filename)

        # Update the `path`
        if "path" in video['result']:
            video["result"]["path"] = "{}\\{}".format(folderToReplace, video_filename)

        if "last_file_path" in video['result']:
            video["result"]["last_file_path"] = "{}\\{}".format(folderToReplace, video_filename)

        # Remove `encodedPath` if needed
        if "encodedPath" in video['result']:
            del video["result"]["encodedPath"]

        newInputStore[items] = video

    return newInputStore


def get_args():
    """
        Returns the filenames from the command line arguments
    """
    return {
        "file": "C:\\Users\\user\\Downloads\\Insights-Capture-1754366039652.bak",# args.input,
        "folderToReplace": "D:\\Videos\\Insights Capture" # args.remap_path,
    }

args = get_args()
backup = read_backup(args["file"])
if not backup:
    print("Could not read backup file: {}".format(args['file']))
    exit()

if not args["folderToReplace"]:
    print("[+] No folderToReplace arg included, dumping converted backup then exiting...")

else:
  print("[+] Updating Video Store to {}".format(args['folderToReplace']))
  videoStore = updateVideoStore(backup['videoStore'], args['folderToReplace'])
  backup['videoStore'] = videoStore

  print("[+] Printing first video's object for manual verification...")
  print(dumps(videoStore[list(videoStore.keys())[0]], indent=2))

output_file = "./backup_dumps/Insights-Capture-{}.json".format(get_current_unixtime())
save_backup("./backup_dumps/Insights-Capture-{}.bak".format(get_current_unixtime()), backup)
save_backup(output_file, backup, "json")


print("[+] Done, saved to {}".format(output_file))
