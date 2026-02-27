#date: 2026-02-27T17:12:25Z
#url: https://api.github.com/gists/8e3b63fb077530dffc2964b648145ec9
#owner: https://api.github.com/users/eshaham

#!/bin/bash
# https://gist.github.com/eshaham/8e3b63fb077530dffc2964b648145ec9
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:$PATH"

[ -z "$CLAUDE_ENV_FILE" ] && exit 0

project_dir="${CLAUDE_PROJECT_DIR:-$(pwd)}"

find_envrc() {
  local dir="$1"
  while [ "$dir" != "/" ]; do
    if [ -f "$dir/.envrc" ]; then
      echo "$dir/.envrc"
      return 0
    fi
    dir=$(dirname "$dir")
  done

  if git -C "$project_dir" rev-parse --show-toplevel >/dev/null 2>&1; then
    local toplevel
    toplevel=$(git -C "$project_dir" rev-parse --show-toplevel 2>/dev/null)
    if [ -f "$toplevel/.envrc" ]; then
      echo "$toplevel/.envrc"
      return 0
    fi

    local common_dir
    common_dir=$(git -C "$project_dir" rev-parse --path-format=absolute --git-common-dir 2>/dev/null)
    if [ -n "$common_dir" ]; then
      local main_repo
      main_repo=$(dirname "$common_dir")
      if [ -f "$main_repo/.envrc" ]; then
        echo "$main_repo/.envrc"
        return 0
      fi
    fi
  fi

  return 1
}

envrc_path=$(find_envrc "$project_dir") || exit 0

env_before=$(env | sort)

set -a
source "$envrc_path"
set +a

env_after=$(env | sort)

new_vars=$(comm -13 <(echo "$env_before") <(echo "$env_after"))

if [ -n "$new_vars" ]; then
  while IFS='=' read -r key value; do
    [ -z "$key" ] && continue
    echo "export $key=\"$value\"" >> "$CLAUDE_ENV_FILE"
  done <<< "$new_vars"
  echo "direnv: loaded $envrc_path"
fi
