#date: 2023-05-17T16:42:52Z
#url: https://api.github.com/gists/7eab5653f8c6d1c6e64d7319a05bc728
#owner: https://api.github.com/users/mettlex

#!/bin/bash

SHORTCUT_PATH="/usr/share/applications/google-chrome.desktop"
LOCAL_SHORTCUT_DIR_PATH="~/.local/share/applications/"
FLAG="--force-dark-mode"

echo "Editing $SHORTCUT_PATH"

# Check if the flag is already present
if grep -n -e "$FLAG" "$SHORTCUT_PATH"; then
    echo "The flag is already present. No changes needed."
else
    # Backup the original file
    sudo cp "$SHORTCUT_PATH" "$SHORTCUT_PATH.bak"

    # Edit the file with the desired changes
    sudo sed -Ei 's|^Exec=/usr/bin/google-chrome-stable(%[a-zA-Z])?(\s.*)?$|Exec=/usr/bin/google-chrome-stable --force-dark-mode\2|' "$SHORTCUT_PATH"

    # Restore the incognito line (if modified by mistake)
    sudo sed -i 's|^Exec=/usr/bin/google-chrome-stable --force-dark-mode --incognito$|Exec=/usr/bin/google-chrome-stable --incognito|' "$SHORTCUT_PATH"

    # Create the directory if it does not exists
    sudo mkdir -p "$LOCAL_SHORTCUT_DIR_PATH"

    # Copy the shortcut to the local shortcut directory
    sudo cp "$SHORTCUT_PATH" "$LOCAL_SHORTCUT_DIR_PATH"

    echo "Flag added successfully."
fi

echo "Done"