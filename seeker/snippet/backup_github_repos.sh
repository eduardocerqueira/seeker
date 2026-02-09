#date: 2026-02-09T17:21:34Z
#url: https://api.github.com/gists/5787c66d7cfdd50a4048d68b7bfa7849
#owner: https://api.github.com/users/ellman12

#!/bin/bash
# Makes local mirrors of all your GitHub repos using the GitHub CLI.

USERNAME=""
PARALLEL_JOBS=8
current_date=$(date +"%Y-%m-%d %I;%M;%S %p")

mkdir -p "$current_date"
cd "$current_date" || exit 1

mapfile -t REPOS < <(gh repo list "$USERNAME" --limit 1000 --json name -q '.[].name')

echo "Found ${#REPOS[@]} repositories for $USERNAME, cloning with $PARALLEL_JOBS jobs"

printf "%s\n" "${REPOS[@]}" \
| xargs -P "$PARALLEL_JOBS" -I {} bash -c '
    echo "Cloning {}"
    gh repo clone "'"$USERNAME"'"/"{}" "{}.git" -- --mirror --quiet
'
