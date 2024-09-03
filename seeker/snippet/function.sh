#date: 2024-09-03T16:48:38Z
#url: https://api.github.com/gists/efda108268ac08bb45984f8bacf965aa
#owner: https://api.github.com/users/mkpanq

# Function to set the latest version of a package locally and install if necessary
asdf_latest_local() {
  local plugin="$1"
  local latest_version

  # Check if a plugin name is provided
  if [[ -z "$plugin" ]]; then
    echo "Usage: asdf_latest_local <plugin>"
    return 1
  fi

  # Get the latest version available for the plugin
  latest_version=$(asdf latest "$plugin")

  # Check if the latest version is already installed
  if ! asdf list "$plugin" | grep -q "$latest_version"; then
    echo "Installing latest version of $plugin: $latest_version"
    asdf install "$plugin" "$latest_version"
  else
    echo "Latest version of $plugin is already installed: $latest_version"
  fi

  # Set the latest version locally
  echo "Setting $plugin to use version $latest_version locally"
  asdf local "$plugin" "$latest_version"
}
