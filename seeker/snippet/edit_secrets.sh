#date: 2024-10-03T17:00:26Z
#url: https://api.github.com/gists/2693fe578afcd4d871c87c5ada685c3d
#owner: https://api.github.com/users/jns

#!/bin/bash

# Check if the filename was provided as an argument
if [ -z "$1" ]; then
  echo "Usage: $0 <filename>"
  exit 1
fi

echo -n "Master Passphrase: "
read -s MASTER_PASSPHRASE

# Variables
ENCRYPTED_FILE="$1"
DECRYPTED_FILE="${ENCRYPTED_FILE%.gpg}.tmp"

if [ -f "$ENCRYPTED_FILE" ]; then
    # Decrypt the file
    gpg --batch --yes --passphrase $MASTER_PASSPHRASE --cipher-algo AES256 --output "$DECRYPTED_FILE" --decrypt "$ENCRYPTED_FILE"
    if [ $? -ne 0 ]; then
    echo "Decryption failed!"
    exit 2
    fi
fi


# Open the decrypted file in vi editor
vi "$DECRYPTED_FILE"

# Re-encrypt the file
gpg --yes --batch --passphrase $MASTER_PASSPHRASE --cipher-algo AES256 --output "$ENCRYPTED_FILE"  --symmetric "$DECRYPTED_FILE"
if [ $? -ne 0 ]; then
  echo "Encryption failed!"
  rm -f "$DECRYPTED_FILE"
  exit 3
fi

# Remove the decrypted temporary file
rm -f "$DECRYPTED_FILE"

echo "File has been successfully edited and re-encrypted."