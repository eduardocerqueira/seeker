#date: 2025-12-18T17:02:00Z
#url: https://api.github.com/gists/8a06b66b769030dd91fa42b24a9bd3cf
#owner: https://api.github.com/users/chaws

#!/bin/bash

# Create a user with the same UID/GID as the host user
# This ensures files created in the container have the correct ownership

USER_NAME="${HOST_USER:-claude-user}"
USER_UID="${HOST_UID:-1000}"
USER_GID="${HOST_GID:-1000}"
USER_HOME="${HOST_HOME:-/home/$USER_NAME}"

# Create group if it doesn't exist
if ! getent group "$USER_GID" > /dev/null 2>&1; then
    groupadd -g "$USER_GID" "$USER_NAME"
fi

# Create home directory first to avoid useradd warnings
mkdir -p "$USER_HOME"

# Create user if it doesn't exist
if ! id "$USER_UID" > /dev/null 2>&1; then
    useradd -u "$USER_UID" -g "$USER_GID" -d "$USER_HOME" -M -s /bin/bash "$USER_NAME"
fi

# Ensure .claude directory and file exist
mkdir -p "$USER_HOME/.claude"
touch "$USER_HOME/.claude.json"

# Set ownership on home directory and all contents
# Do this explicitly for the directory itself and its contents
chown "$USER_UID:$USER_GID" "$USER_HOME"
chown -R "$USER_UID:$USER_GID" "$USER_HOME/.claude"
chown "$USER_UID:$USER_GID" "$USER_HOME/.claude.json"

# Ensure the user has access to workspace
chown -R "$USER_UID:$USER_GID" /workspace 2>/dev/null || true

# Make sure home directory and config files are writable
chmod 755 "$USER_HOME"
chmod -R u+w "$USER_HOME/.claude"
chmod u+w "$USER_HOME/.claude.json"

# Switch to the user and run Claude Code
exec sudo -u "#$USER_UID" -g "#$USER_GID" HOME="$USER_HOME" claude "$@"
