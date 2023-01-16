#date: 2023-01-16T17:01:54Z
#url: https://api.github.com/gists/c4da21af315b62fc3b88163541c816e3
#owner: https://api.github.com/users/eugeneyan

"""
Iteratively loop through all files in DIR and add-commit-push them to REPO
"""
from pathlib import Path
from git import Repo
import os

DIR = '/Users/eugene/obsidian-vault/assets'
REPO = Repo('.')

for i, path in enumerate(Path(DIR).iterdir()):
    file_size = int(os.stat(path).st_size / 1024)
    print(f'file {i}: {str(path).split("/")[-1]} ({file_size} kb)')

    if file_size < 1024:

        commit_msg = f'Add {str(path).split("/")[-1]}'

        REPO.index.add(str(path))  # Add
        REPO.index.commit(commit_msg)  # Commit
        origin = REPO.remote('origin')
        origin.push()  # Push

    else:
        print(f'Skipping file {i}: {str(path).split("/")[-1]} ({file_size} kb)')

        f = open("skipped-files.txt", "a")
        f.write(f'{str(path).split("/")[-1]}\n')
        f.close()
