#date: 2024-08-26T16:46:56Z
#url: https://api.github.com/gists/1fa4ff0c1e664cf3c6949a42b1fa6811
#owner: https://api.github.com/users/flaksp

import re

CODEOWNERS_FILE_PATH = "CODEOWNERS"
PROJECT_NAME = "my-project-name"
OWNERS_TO_SEARCH_FOR = {
    "my-codeowners-team-name": True,
    "my-another-codeowners-team-name": True,
}

scope_lines = []


def codeowners_path_to_scope_line(file_path):
    # Remove leading /
    file_path = re.sub(r'^/', '', file_path)

    # Replace trailing / with //**
    if file_path.endswith('/'):
        file_path += '/**'

    return 'file[' + PROJECT_NAME + ']:' + file_path


with open(CODEOWNERS_FILE_PATH) as file:
    for line in file:
        if not line.startswith('/'):
            continue

        path_and_team = re.split(r' +@', line, 2)

        path = path_and_team[0].strip()
        owners = re.split(r' +@', path_and_team[1].strip())

        for owner in owners:
            if owner in OWNERS_TO_SEARCH_FOR:
                scope_lines.append(codeowners_path_to_scope_line(path))
                break
        else:
            continue

print("\n".join(scope_lines))