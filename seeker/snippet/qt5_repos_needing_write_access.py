#date: 2023-01-13T17:04:55Z
#url: https://api.github.com/gists/58baedbed87bc7a1abf96803ec852ad6
#owner: https://api.github.com/users/keithel

#!/usr/bin/env python
import sys
import os
import os.path
import requests

if __name__ == "__main__":

    if not os.path.isfile("./init-repository"):
        print("Please execute from the base of a qt5 repository", file=sys.stderr)
        exit(1)

    try:
        token= "**********"
    except KeyError:
        print("GITHUB_TOKEN not set. aborting.", file= "**********"
        exit(1)

    dir_entries = os.scandir()
    candidate_repos=[e for e in dir_entries if (e.name.startswith("qt") and os.path.isdir(e.name))]

    removed_candidates=[e for e in dir_entries if (e.name.startswith("qt") and not os.path.isdir(e.name))]
    if len(removed_candidates) > 0:
        print("The following candidate repos were removed:")
        for candidate in removed_candidates:
            print(f"    {candidate.name}")

    candidate_repo_json={}
    for repo in candidate_repos:
        sys.stdout.write(".")
        sys.stdout.flush()
        url=f"https://git.example.com/api/v3/repos/oss-forks/{repo.name}"
        response=requests.get(url, headers={"Authorization": "**********"
        candidate_repo_json[repo] = response
        try:
            permissions=response['permissions']
            if permissions['push'] != True:
                try:
                    repos_needing_write.append(repo)
                except NameError:
                    repos_needing_write=[]
                    repos_needing_write.append(repo)
        except KeyError:
            try:
                repos_needing_creation.append(repo)
            except NameError:
                repos_needing_creation=[]
                repos_needing_creation.append(repo)
    print()

    try:
        repos_needing_write
        print("Write/Push permissions to the following qt5 submodule repositories are needed:")
        for repo in repos_needing_write:
            print(f"    {repo.name}")
    except NameError:
        print("Write/Push permission to all needed qt submodule repositories is in place.")

    print()
    try:
        repos_needing_creation
        print("The following qt5 submodule repositories do not exist in oss-forks.")
        print("They need creating, and I need write/push permissions to them:")
        for repo in repos_needing_creation:
            print(f"    {repo.name}")
    except NameError:
        print("Creation of all missing qt submodule repositories has been done.")

    if not 'repos_needing_write' in locals() and not 'repos_needing_creation' in locals():
        print("Manual mirroring of all qt5 6.5 submodule repos should be possible from kyzik account now.")
rom kyzik account now.")
