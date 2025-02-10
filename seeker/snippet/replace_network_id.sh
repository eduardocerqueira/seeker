#date: 2025-02-10T16:47:10Z
#url: https://api.github.com/gists/a3fc0471885909f9e61f21de5cdef98d
#owner: https://api.github.com/users/vanviethieuanh

#!/bin/bash
# This bash script replace all OLD_NETWORK_ID with NEW_NETWORK_ID of all containers
# You might want to run this script with sudo
# Remember to restart docker after run this script.

OLD_NETWORK_ID="your_old_network_id"
NEW_NETWORK_ID="your_new_network_id"

# Iterate through each container's config.v2.json file
find /$DOCKER_HOME/containers/*/config.v2.json -type f | while read config_file; do
  # Replace OLD_NETWORK_ID with NEW_NETWORK_ID in each file
  sed -i "s/$OLD_NETWORK_ID/$NEW_NETWORK_ID/g" "$config_file"
done
