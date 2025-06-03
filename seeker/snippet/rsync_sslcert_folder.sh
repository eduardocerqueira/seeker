#date: 2025-06-03T16:57:39Z
#url: https://api.github.com/gists/3c2fc5f96431d8e7255a4050187e341f
#owner: https://api.github.com/users/alistairhendersoninfo

#!/bin/bash
set -e

read -rp "Enter remote server IP: " REMOTE_IP

SRC_BASE="/etc/letsencrypt/live"
DEST_BASE="/etc/letsencrypt/live"
KEY_PATH="$HOME/.ssh/certsync"
META_BASE="/opt/sslcopy"

# Ensure metadata dir exists
sudo mkdir -p "$META_BASE"

# Loop through domain dirs
for DIR in "$SRC_BASE"/*; do
  [ -d "$DIR" ] || continue  # Skip if not a directory

  DIRNAME=$(basename "$DIR")
  METAFILE="$META_BASE/$DIRNAME"

  echo "🔁 Syncing $DIR to $REMOTE_IP..."

  rsync -avz -e "ssh -i $KEY_PATH" "$DIR" "sslcert@$REMOTE_IP:$DEST_BASE/"

  echo "IP=$REMOTE_IP" | sudo tee "$METAFILE" > /dev/null
  echo "DIR=$DIR" | sudo tee -a "$METAFILE" > /dev/null

  echo "✅ Synced $DIR → $REMOTE_IP and wrote $METAFILE"
done
