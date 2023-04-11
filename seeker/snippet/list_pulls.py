#date: 2023-04-11T16:49:57Z
#url: https://api.github.com/gists/0336ffa19e1d40254deb317c0224a264
#owner: https://api.github.com/users/bkueng

#!/usr/bin/env python3
""" Script to create a list of pull requests between 2 commits """

# generate token: "**********"://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token

import argparse
import subprocess
import requests
import json

github_repo = 'PX4/PX4-Autopilot'

parser = argparse.ArgumentParser(description='Script to create a list of pull requests between 2 commits')

parser.add_argument('a', help='Base commit sha')
parser.add_argument('b', help='Head commit sha')
parser.add_argument('--repo-dir', '-r', default='.', help='Repository directory (use cwd if not given)')
parser.add_argument('--token', '-t', help= "**********"

args = parser.parse_args()

a = args.a
b = args.b
repo_dir = args.repo_dir
token = "**********"

cmd='git rev-list --ancestry-path {:}..{:}'.format(a, b)
commits = subprocess.check_output(cmd.split(), cwd=repo_dir).decode('utf-8').split('\n')
commits.append(a)
commits.reverse()

found_pulls = set()

for commit in commits:
    if commit == '':
        continue
    url = 'https://api.github.com/repos/{:}/commits/{:}/pulls'.format(github_repo, commit)
    headers={"Accept": "application/vnd.github.v3+json"}
 "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"  "**********"i "**********"s "**********"  "**********"n "**********"o "**********"t "**********"  "**********"N "**********"o "**********"n "**********"e "**********": "**********"
        headers["Authorization"] = "**********"
    response = requests.get(url, headers=headers)
    try:
        assert response.ok
        response_json = json.loads(response.content)
        assert(len(response_json) <= 1)
        if len(response_json) == 0:
            # seems to be a push-to-master w/o PR
            print('commit: {:}'.format(commit))
        else:
            url = response_json[0]['html_url']
            pr_number = response_json[0]['number']
            title = response_json[0]['title']
            merged_at = response_json[0]['merged_at']
            if pr_number not in found_pulls:
                found_pulls.add(pr_number)
                print('{:}, {:}, {:}'.format(url, pr_number, title))
    except AssertionError as e:
        print("Exception for:")
        print(url)
        print(response.status_code)
        print(response.content)
        raise e
