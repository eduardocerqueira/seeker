#date: 2023-08-31T17:00:19Z
#url: https://api.github.com/gists/f638ba6a6208e4e37f49ccae94cc948e
#owner: https://api.github.com/users/eggplants

#!/usr/bin/env bash

set -euxo pipefail

if [[ $# == 0 ]]
then
  echo "$0 FILE_ID [SAVE_PATH]"
  exit 0
fi

if ! command -v wget &>/dev/null
then
  echo "install: wget" >&2
  exit 1
fi

file_id="$1"
save_path="${2:-.}"

if [[ -z "$file_id" ]]
then
  echo "FILE_ID seems to be empty, aight?" >&2
  exit 1
fi

if ! [[ -e "$save_name" ]]
then
  echo "File already exists. Overwrite?([y]:n)" >&2
  read yn
  if [[ "$yn" =~ ^n ]]
  then
    exit 0
  fi
fi

u="https://docs.google.com/uc?export=download&id=${file_id}"
cs="$(
  wget \
    --quiet \
    --keep-session-cookies \
    --no-check-certificate \
    --save-cookies /tmp/cookies.txt "$u" -O- |
  sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p'
)"
eval wget --load-cookies /tmp/cookies.txt "${u}&confirm=${cs}" \
          "$([[ -n "$save_path" ]] && echo -- '-O "$save_path"')"

echo "[DONE]"