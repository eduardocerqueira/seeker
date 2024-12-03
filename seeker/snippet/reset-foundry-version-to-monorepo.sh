#date: 2024-12-03T16:55:41Z
#url: https://api.github.com/gists/54ff275b32482823029816d9f1e29a9c
#owner: https://api.github.com/users/blmalone

resetFoundryVersion() {
  local TOML_URL="https://raw.githubusercontent.com/ethereum-optimism/optimism/refs/heads/develop/mise.toml"

  local TOML_FILE="/tmp/mise.toml"
  curl -s -o "$TOML_FILE" "$TOML_URL"

  if [[ ! -f "$TOML_FILE" ]]; then
    echo "Failed to download the TOML file."
    return 1
  fi

  if command -v yq &>/dev/null; then
    local forge_version=$(yq e '.tools.forge' "$TOML_FILE")
  else
    echo "yq is not installed. Install it by running: brew install yq"
    return 1
  fi

  if [[ -z "$forge_version" ]]; then
    echo "Failed to extract the forge version from the TOML file."
    return 1
  fi

  echo "Forge version: $forge_version"

  foundryup -C "$forge_version"

  if [[ $? -ne 0 ]]; then
    echo "Failed to run the foundryup command."
    return 1
  fi

  echo "Foundry updated successfully with version: $forge_version"
}