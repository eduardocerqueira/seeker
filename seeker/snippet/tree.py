#date: 2023-02-27T17:05:59Z
#url: https://api.github.com/gists/0b9b3c003766f5bc9323cdcb925afafa
#owner: https://api.github.com/users/ivanistheone

#!/usr/bin/env python
import argparse
import os

def tree(directory=".", indent = "    "):
    """
    Helper function that prints the filesystem tree.
    """
    ignorables = ["__pycache__", ".gitignore", ".DS_Store", ".ipynb_checkpoints", ".git", "venv"]
    for root, dirs, files in os.walk(directory):
        path = root.split(os.sep)
        if any(ign in path for ign in ignorables):
            continue
        print((len(path) - 1) * indent, os.path.basename(root) + "/")
        for file in files:
            if file in ignorables:
                continue
            print(len(path) * indent, file)
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog = 'tree', description = 'list contents of directories in a tree-like format.')
    parser.add_argument('directory', nargs='?', default=".")
    args = parser.parse_args()
    tree(args.directory)
