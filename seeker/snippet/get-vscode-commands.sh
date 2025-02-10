#date: 2025-02-10T17:07:48Z
#url: https://api.github.com/gists/08ac86fc0cc72817d6c715b759961c82
#owner: https://api.github.com/users/tgran2028

#!/bin/bash

# manually save the default keybindings to a file
defaults="$TEMP/keybindings.vscode-default.jsonc"

mapfile -t used_commands < <(
  jsonc read < "$defaults" |
    jq -M '.[].command' |
    xargs -n1
)

mapfile -t unused_commands < <(
  grep -E '^[[:space:]]*\/\/[[:space:]]*-[[:space:]]+[A-Za-z0-9_.-]+' "$defaults" |
    sed -E 's/^[[:space:]]*\/\/[[:space:]]*-[[:space:]]+//' |
    xargs -n1d 
)

declare -a commands=( "${used_commands[@]}" "${unused_commands[@]}" )

# Capture commands as JSON array
commands_json=$(printf "%s\n" "${commands[@]}" | grep -v '^$' | jq -R . | jq -s .)

# Build extensions JSON array:
# This command produces one JSON object per extension, assuming that each non-empty line
# from code-insider --list-extensions --show-versions consists of two fields: id and version.
extensions_json=$(code --list-extensions --show-versions | awk 'NF {print "{\"id\":\"" $1 "\",\"version\":\"" $2 "\"}"}' | jq -s .)

# Build the final JSON object with metadata and the commands array.
jq -n \
  --arg vscode "code" \
  --arg current_datetime "$(date +'%Y-%m-%dT%H:%M:%S%z')" \
  --arg version "$(code --version | head -n1)" \
  --argjson extensions "$extensions_json" \
  --argjson commands "$commands_json" \
  '{vscode: $vscode, current_datetime: $current_datetime, version: $version, extensions: $extensions, commands: $commands}' >| commands.json

