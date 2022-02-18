#date: 2022-02-18T17:07:44Z
#url: https://api.github.com/gists/2ab127085052a8d78673b4e4ed6c5931
#owner: https://api.github.com/users/dcaldr

import shutil
import os
import sys


# Temporary increase the recursion limit to handle deep directory structures and reverts it back to the original value
class MyRecursionLimit:
    def __init__(self, limit):
        self.limit = limit

    def __enter__(self):
        self.old_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(self.limit)

    def __exit__(self, type, value, tb):
        sys.setrecursionlimit(self.old_limit)


# User input for the path to the folder to be deleted
path = input("Enter the path to the folder to be deleted: ")


# remove filename too long folder permissions and delete folder
def delete_folder_and_permissions(folder):
    # remove folder permissions
    os.chmod(folder, 0o777)
    # delete folder
    shutil.rmtree(folder)
    print("Deleted folder: " + folder)


# delete folder recursively
with MyRecursionLimit(100000):
    delete_folder_and_permissions(path)
