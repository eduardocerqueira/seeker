#date: 2025-04-29T17:13:18Z
#url: https://api.github.com/gists/2f230af88f81b71f17863a24d5a63eac
#owner: https://api.github.com/users/dolohow

import subprocess
import os
import sys

# === CONFIGURATION ===
PREFERRED_DIR = ""
TARGET_DIRS = ["", ""]  # directories to scan with fdupes
DRY_RUN = True  # Set to False to actually delete files

def get_fdupes_output(dirs):
    try:
        result = subprocess.run(["fdupes", "-r"] + dirs, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print("Error running fdupes:", e)
        sys.exit(1)

def parse_duplicate_groups(fdupes_output):
    groups = []
    current_group = []

    for line in fdupes_output.strip().split("\n"):
        line = line.strip()
        if line == "":
            if len(current_group) > 1:
                groups.append(current_group)
            current_group = []
        else:
            current_group.append(line)

    if len(current_group) > 1:
        groups.append(current_group)

    return groups

def delete_duplicates(groups, preferred_dir):
    for group in groups:
        keep = None
        to_delete = []

        for file in group:
            if keep is None and file.startswith(preferred_dir):
                keep = file
            else:
                to_delete.append(file)

        if keep:
            print(f"\nKeeping: {keep}")
            for file in to_delete:
                print(f"Deleting: {file}")
                if not DRY_RUN:
                    try:
                        os.remove(file)
                    except Exception as e:
                        print(f"Failed to delete {file}: {e}")
        else:
            print(f"\n⚠️ No preferred file found in group: {group}")
            print("Skipping deletion to be safe.")

def main():
    print("Scanning for duplicate files...")
    output = get_fdupes_output(TARGET_DIRS)
    groups = parse_duplicate_groups(output)
    print(f"Found {len(groups)} groups of duplicates.")

    delete_duplicates(groups, PREFERRED_DIR)

if __name__ == "__main__":
    main()
