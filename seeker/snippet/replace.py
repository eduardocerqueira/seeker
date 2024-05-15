#date: 2024-05-15T16:54:03Z
#url: https://api.github.com/gists/9eea494be56b4a0b04c4ac7d5a6e1292
#owner: https://api.github.com/users/Frank-Buss

#!/usr/bin/env python3

# automatically pins all your workflows, see
# https://github.com/fmtlib/fmt/issues/3449
# for details

import sys
import requests
import os

 "**********"d "**********"e "**********"f "**********"  "**********"g "**********"e "**********"t "**********"_ "**********"g "**********"i "**********"t "**********"h "**********"u "**********"b "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"( "**********") "**********": "**********"
    # get your github token here, like from env variable
    return None

 "**********"d "**********"e "**********"f "**********"  "**********"g "**********"e "**********"t "**********"_ "**********"c "**********"o "**********"m "**********"m "**********"i "**********"t "**********"_ "**********"h "**********"a "**********"s "**********"h "**********"( "**********"r "**********"e "**********"p "**********"o "**********", "**********"  "**********"r "**********"e "**********"f "**********"e "**********"r "**********"e "**********"n "**********"c "**********"e "**********", "**********"  "**********"g "**********"i "**********"t "**********"h "**********"u "**********"b "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"= "**********"N "**********"o "**********"n "**********"e "**********") "**********": "**********"
    headers = {}
 "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"g "**********"i "**********"t "**********"h "**********"u "**********"b "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********": "**********"
        headers['Authorization'] = "**********"

    # First, try to get the commit hash for the tag
    tag_url = f"https://api.github.com/repos/{repo}/git/ref/tags/{reference}"
    tag_response = requests.get(tag_url, headers=headers)
    if tag_response.status_code == 200:
        return tag_response.json()['object']['sha']
    else:
        # If tag is not found, try to get the commit hash for the branch
        branch_url = f"https://api.github.com/repos/{repo}/git/ref/heads/{reference}"
        branch_response = requests.get(branch_url, headers=headers)
        if branch_response.status_code == 200:
            return branch_response.json()['object']['sha']
        else:
            print(f"Error fetching commit hash for {repo}@{reference}: {branch_response.status_code}")
            return None

 "**********"d "**********"e "**********"f "**********"  "**********"r "**********"e "**********"p "**********"l "**********"a "**********"c "**********"e "**********"_ "**********"t "**********"a "**********"g "**********"s "**********"_ "**********"w "**********"i "**********"t "**********"h "**********"_ "**********"h "**********"a "**********"s "**********"h "**********"e "**********"s "**********"( "**********"f "**********"i "**********"l "**********"e "**********"_ "**********"p "**********"a "**********"t "**********"h "**********", "**********"  "**********"g "**********"i "**********"t "**********"h "**********"u "**********"b "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"= "**********"N "**********"o "**********"n "**********"e "**********") "**********": "**********"
    with open(file_path) as file:
        lines = file.readlines()

    updated_lines = []
    for line in lines:
        line = line
        if 'uses: ' in line:
            parts = line.split('uses: ')[1].split('@')
            if len(parts) == 2:
                repo, tag = parts
                if '/' in repo:
                    tag = tag.strip().split(' ')[0]
                    if len(tag) < 40:
                        commit_hash = "**********"
                        if commit_hash:
                            line2 = line.replace(f"{repo}@{tag}", f"{repo}@{commit_hash} # {tag}")
                            print(f"org: {line.strip()}")
                            print(f"replaced: {line2.strip()}")
                            line = line2
        updated_lines.append(line)

    with open(file_path, "w") as file:
        file.writelines(updated_lines)

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 script.py <path_to_yaml_file_1> <path_to_yaml_file_2> ...")
        sys.exit(1)

    github_token = "**********"

    for file_path in sys.argv[1:]:
        print(f"Processing {file_path}...")
        replace_tags_with_hashes(file_path, github_token)

if __name__ == "__main__":
    main()