#date: 2021-12-17T16:57:33Z
#url: https://api.github.com/gists/cd581f36b2b87fd00f57562dbf6f32b8
#owner: https://api.github.com/users/aparkerlue

# -*- mode: shell-script; sh-shell: bash; coding: utf-8; -*-
git-tree-files() {
    git ls-files "$@" | tree -a --fromfile .
}
