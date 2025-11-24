#date: 2025-11-24T16:59:18Z
#url: https://api.github.com/gists/92c07bfd6408428c6955a868bbdd72f4
#owner: https://api.github.com/users/ElliotFriend

import argparse
import requests

from typing import TypedDict

parser = argparse.ArgumentParser(
    prog="MutationsTester",
    description="Check for actual repos in a given mutation file",
    epilog="Don't try this at home, kids",
)
parser.add_argument("filename",
                    help="the path to file which should be parsed and tested")
parser.add_argument("-c", "--count",
                    action="store_true",
                    help="count the number of good/bad repos in the mutation")
parser.add_argument("-u", "--update",
                    action="store_true",
                    help="update the mutations file in-place, removing the errored repos")
parser.add_argument("-v", "--verbose",
                    action="store_true",
                    help="list all repos and statuses found in mutation")
args = parser.parse_args()

filename: str = args.filename
should_count: bool = args.count
should_update: bool = args.update
is_verbose: bool = args.verbose

GITHUB_TOKEN: "**********"

class CountInfo(TypedDict):
    ok: int
    error: int

count: CountInfo = {
    "ok": 0,
    "error": 0,
}
error_repos: set[tuple[int, str]] = set()

with open(filename, "r") as file:
    lines = file.readlines()
    for line in lines:
        # assuming that EVERY line is `repadd <ecosystem> repo_url`
        [_, _, repo_url] = line.strip().split(' ')
        [owner, repo] = repo_url.split('/')[-2:]

        if is_verbose:
            print(f"Testing URL: {repo_url}")

        # api_url = repo_url.replace("https://", "https://api.")
        api_url = f"https://api.github.com/repos/{owner}/{repo}"
        response = requests.get(api_url, headers={
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "Authorization": "**********"
        })

        if response.ok:
            # print a message
            if is_verbose:
                print(f"\tOK {response.status_code} for {repo_url}")

            # add to count
            if should_count:
                count["ok"] += 1

        else:
            # print a message
            if is_verbose:
                print(f"\tERROR {response.status_code} for {repo_url} (api: {api_url})")

            # add to count
            if should_count:
                count["error"] += 1

            # add error repo to the set, along with status code
            error_repos.add((response.status_code, repo_url))

if should_count:
    print(f"ok repos: {count['ok']}, error repos: {count['error']}")

if len(error_repos) > 0:
    # spit out the error'd
    print(f"Errors found:\n{'\n'.join(map(str, [f'\t{er}' for er in list(error_repos)]))}")

    if should_update:
        bad_repo_urls = [er[1] for er in error_repos]
        with open(filename, "w") as file:
            for line in lines:
                # again, assuming that EVERY line is `repadd <ecosystem> repo_url`
                line_url = line.strip().split(' ')[-1]
                if line_url not in bad_repo_urls:
                    file.write(line)

else:
    print("No errors found")

                    file.write(line)

else:
    print("No errors found")

