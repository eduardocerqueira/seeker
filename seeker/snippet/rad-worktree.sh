#date: 2024-05-07T16:53:19Z
#url: https://api.github.com/gists/256eb5b4ddaf8ec55d9ca3db3e736f86
#owner: https://api.github.com/users/lorenzleutgeb

#! /bin/sh
set -eu

jq -V >/dev/null

if [ $# -lt 1 ]
then
    printf 'Usage: %s rad:… [name-of-worktree]\n' "$0"
    printf 'The worktree will be created in a new directory within the current working directory.'
    exit 1
fi

RID="$1"
BARE="${RAD_HOME:-$HOME/.radicle}/storage/${RID:4}"
if [ "${RID:0:4}" != "rad:" ]
then
    printf "Expected RID '%s' to start with 'rad:'. Aborting." "$RID"
    exit 2
elif [ ! -d "$BARE" ]
then
    printf "Expected '%s' to exist (maybe invoke \`rad seed\`?). Aborting." "$BARE"
    exit 3
fi
RID="${RID:4}"

NAME="${2:-$(rad inspect --identity "$RID" | jq -r '.payload."xyz.radicle.project".name')}"

git -C "$BARE" worktree add "$PWD/$NAME"

printf "\nYou will very likely want to set up the \`rad\` remote in your worktree,\n"
printf "especially if this is the first worktree you create for this repo.\n\n"
printf "  1. Change directory to your newly created worktree.\n"
printf "         cd %s\n\n" "$NAME"
printf "  2. Execute the following two commands.\n"
printf "         git remote add rad rad://%s\n" "${RID}"
printf "         git remote set-url rad --push rad://%s/%s\n\n" "${RID}" "$(rad self --nid)"