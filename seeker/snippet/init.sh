#date: 2024-01-30T17:02:23Z
#url: https://api.github.com/gists/d50615cb3358dc1c55ae37bf8c566f10
#owner: https://api.github.com/users/wastrachan

#!/bin/sh

# Check for required project dependencies and install
# git hooks bundled with this project

echo ""
echo "Checking dependencies..."
if [[ -z "$(command -v git-secret)" ]]; then
    cat <<\EOF

Error: "**********"

It is not possible to lock up or reveal secret files. Ensure that git-secret
(https: "**********"

EOF
    exit 1
fi

if [[ -z "$(command -v docker)" ]]; then
    cat <<\EOF

Error: docker is not installed

Docker is required to run this project. Ensure that docker
(https://www.docker.com/) is installed and available on your path.

EOF
    exit 1
fi

echo "Copying git hooks into .git/hooks..."
HOOK_SOURCE="$(git rev-parse --show-toplevel)/.scripts/git-hooks"
HOOK_DIR="$(git rev-parse --show-toplevel)/.git/hooks"
HOOK_NAMES="pre-commit post-update"
for hook in $HOOK_NAMES; do
  rm -rf $HOOK_DIR/$hook
  ln -s -f $HOOK_SOURCE/$hook $HOOK_DIR/$hook
done
f $HOOK_DIR/$hook
  ln -s -f $HOOK_SOURCE/$hook $HOOK_DIR/$hook
done
