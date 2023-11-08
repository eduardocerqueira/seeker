#date: 2023-11-08T16:48:45Z
#url: https://api.github.com/gists/7ff62f6be144bf095f580e1541a38d98
#owner: https://api.github.com/users/FelixSchladt

# Copyright 2022 Felix Schladt (https://github.com/FelixSchladt)

import json
import sys
import os


def get_files_from_folder(folder):
    try:
        files = os.listdir(folder)
        return [ f"{folder}/{file}" for file in files ]
    except FileNotFoundError:
        print(f"no such file or directory: {folder}")
        sys.exit(-1)
    except NotADirectoryError:
        return [ folder ]


def analyze_files(files):
    vehicles = []
    for file in files:
        try:
            json_data = json.load(open(file))
            for vehicle in json_data["data"]:
                if vehicle["attributes"]["zoneId"] != "FRIEDRICHSHAFEN":
                    continue
                vid = vehicle["id"]
                if not vid in vehicles:
                    vehicles.append(vid)
        except json.JSONDecodeError:
            print("Hi Data Henrik, your data ain't good! :D")
            print(f"Corrupted file: {file}\n{open(file).read()}")
    return len(vehicles)


def main():
    if len(sys.argv) < 2:
        print("Please specify folder to read data from")
        sys.exit(-1)
    files = get_files_from_folder(sys.argv[1])
    print(f"Vehicles with unique IDs in the zone Friedrichshafen:\n{analyze_files(files)}")

if __name__ == "__main__":
    main()