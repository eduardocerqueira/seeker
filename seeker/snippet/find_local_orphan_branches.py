#date: 2023-11-22T16:41:13Z
#url: https://api.github.com/gists/dd52be87ea0104d6a712fcbdba7b71bf
#owner: https://api.github.com/users/hartwork

# Finds Git branches that were tracking a remote branch that no longer exists today.
# For Python >=3.8, all else is end-of-life.
#
# Copyright (c) 2023 Sebastian Pipping <sebastian@pipping.org>
# Licensed under GPL v3 or later

import re
from subprocess import check_output

# NOTE: See "man git-check-ref-format" for colon (":") being disallowed in references
git_branch_output = check_output(['git', 'branch', '-a', '--format=%(refname):%(upstream)'], text=True)

upstream_of_refname = dict(
    # %(refname) format (Git 2.43.0):  refs/heads/<branch_name>
    # %(upstream) format (Git 2.43.0): refs/remotes/<remote>/<branch_name>
    entry.split(':')
    for entry
    in git_branch_output.split('\n')
    if entry
)

for refname, upstream in upstream_of_refname.items():
    expected_prefix = 'refs/heads/'
    if not refname.startswith(expected_prefix):
        continue

    if upstream not in upstream_of_refname:
        branch_name = refname[len(expected_prefix):]
        print(branch_name)
