#date: 2025-03-26T17:13:09Z
#url: https://api.github.com/gists/64e73f8d3de1cf4ca32dd536cb71f63f
#owner: https://api.github.com/users/bovlb

#!/bin/env python3

"""
This script pins the versions of GitHub Actions in a workflow YAML file to the latest stable release.
It reads the workflow file, identifies the GitHub Actions used, and updates their versions to the latest stable release.

Use as: pin_github_actions.py .github/workflows/*.yml
"""

import os
import re
import sys
import requests
from functools import cache

SAFE_OWNER_LIST = [".", "actions", "aws-actions", "docker",
                   "google-github-actions", "hashicorp", "luisremis"]


@cache
def get_pinned_action(action: str):
    """Given an action of the form OWNER/REPO or OWNER/REPO@VERSION, 
    return a version of the action pinned to a full commit hash.
    If the action is already pinned to a specific commit, return it as is.
    Does not change actions with owners on the SAFE_OWNER_LIST.
    """
    if not action or not isinstance(action, str):
        return action
    if "@" in action:
        owner_repo, version = action.split("@", 1)
    else:
        version = None
        owner_repo = action
    owner, repo = owner_repo.split("/", 1)
    if owner in SAFE_OWNER_LIST:
        return action  # Do not change actions from SAFE_OWNER_LIST
    if version and re.match(r"^[0-9a-f]{40}$", version):
        return action  # Already pinned to a full commit hash
    if version and version not in ["latest", "release"]:
        # If the action is already pinned to a version, we need to get that specific release
        # and pin it to a full commit hash.
        url = f"https://api.github.com/repos/{owner_repo}/git/refs/tags/{version}"
        response = requests.get(url)
        response.raise_for_status()
        if response.status_code == 200:
            tag_data = response.json()
            if isinstance(tag_data, dict) and 'object' in tag_data and 'sha' in tag_data['object']:
                return f"{owner_repo}@{tag_data['object']['sha']}"
            else:
                print(
                    f"Warning: No commit hash found for {owner_repo} version {version}")
        else:
            print(
                f"Warning: Unable to fetch {version} release for {owner_repo}")
    else:  # if no version, then get commit hash for latest stable release
        url = f"https://api.github.com/repos/{owner_repo}/git/refs/tags"
        response = requests.get(url)
        response.raise_for_status()
        if response.status_code == 200:
            tags_data = response.json()
            if isinstance(tags_data, list) and len(tags_data) > 0:
                # Sort tags to find the latest stable release
                tags_data.sort(key=lambda x: x['ref'], reverse=True)
                for tag in tags_data:
                    if re.match(r"^v?\d+\.\d+\.\d+$", tag['ref']):
                        # Found a stable release tag
                        sha = tag['object']['sha']
                        return f"{owner_repo}@{sha}"
            else:
                print(f"Warning: No tags found for {owner_repo}")
        else:
            print(f"Warning: Unable to fetch tags for {owner_repo}")
        return action


def pin_actions_in_workflow(workflow_file: str):
    """Read the workflow file, pin all actions to their latest stable release."""
    with open(workflow_file, 'r') as file:
        content = file.read()

    # Find all actions in the workflow file
    action_pattern = r"uses:\s*([a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+(?:@[\w.-]+)?)"
    matches = re.findall(action_pattern, content)

    if not matches:
        print(f"No actions found in {workflow_file}")
        return

    # Pin each action to its latest stable release
    for action in matches:
        try:
            pinned_action = get_pinned_action(action)
        except:
            print(f"Error pinning action {action} in {workflow_file}")
            raise

        if pinned_action != action:
            content = content.replace(action, pinned_action)
            print(f"Updated {action} to {pinned_action}")

    # Write the updated content back to the workflow file
    with open(workflow_file, 'w') as file:
        file.write(content)
    print(f"Updated workflow file: {workflow_file}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: pin_github_actions.py .github/workflows/*.yml")
        sys.exit(1)
    workflow_files = sys.argv[1:]
    for workflow_file in workflow_files:
        if os.path.isfile(workflow_file):
            pin_actions_in_workflow(workflow_file)
        else:
            print(f"File not found: {workflow_file}")
            sys.exit(1)
    print("All workflow files processed.")
    sys.exit(0)
# Note: This script requires the requests library to be installed.
