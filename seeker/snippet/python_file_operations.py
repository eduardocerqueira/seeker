#date: 2025-03-05T16:52:40Z
#url: https://api.github.com/gists/0f82f547f2b51b6d30b606f0d333dbe6
#owner: https://api.github.com/users/fikebr

import os
import shutil

def write_file(file, text):
    f = open(file, "w", encoding="utf-8")
    f.write(text)
    f.close()


def write_file_append(file, text):
    f = open(file, "a", encoding="utf-8")
    f.write(text)
    f.close()


def copy_file(source, dest):
    try:
        shutil.copyfile(source, dest)
    except OSError as e:
        raise (e)


def move_file(source, dest):
    try:
        shutil.move(source, dest)
    except OSError as e:
        raise (e)


def read_file(filename):
    if os.path.exists(filename):
        try:
            with open(filename, "r", encoding="utf-8") as f:
                contents = f.read()
            return contents
        except FileNotFoundError:
            # Handle the case where the file is not found
            print(f"Error: File '{filename}' not found.")
            return None
    else:
        print(f"Error: File '{filename}' does not exist.")
        return None


