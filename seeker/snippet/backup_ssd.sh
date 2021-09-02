#date: 2021-09-02T17:11:35Z
#url: https://api.github.com/gists/20ed640417f6b378f91d23703f60bfa6
#owner: https://api.github.com/users/thiagozs

#! /bin/sh

# set -e

# function dismount()
# {
#   if [ -d "\$mount_point" ]; then
#     veracrypt --text --dismount "\$mount_point"
#   fi
# }

# trap dismount ERR INT

# volume_path="$BACKUP_VOLUME_PATH"
mount_point="/Volumes/KINGDIAN"

# veracrypt --text --mount --pim "0" --keyfiles "" --protect-hidden "no" "\$volume_path" "\$mount_point"

mkdir -p "$mount_point/Versioning"

files=(
  "/Users/$USER/.gnupg"
  "/Users/$USER/.ssh"
  "/Users/$USER/Library/Keychains"
  "/Users/$USER/Tools"
  "/Users/$USER/Temp"
  "/Users/$USER/workspace"
)

for file in "${files[@]}"; do
  rsync \
    -axRS \
    --progress \
    --backup \
    --backup-dir \
    "$mount_point/Versioning" \
    --delete \
    --suffix="$(date +".%F-%H%M%S")" \
    "$file" \
    "$mount_point"
done

if [ "\$(find "$mount_point/Versioning" -type f -ctime +90)" != "" ]; then
  printf "Do you wish to prune versions older than 90 days (y or n)? "
  read -r answer
  if [ "$answer" = "y" ]; then
    find "$mount_point/Versioning" -type f -ctime +90 -delete
    find "$mount_point/Versioning" -type d -empty -delete
  fi
fi

open "$mount_point"

printf "Inspect backup and press enter"

read -r answer

#dismount

printf "Generate hash (y or n)? "
read -r answer
if [ "$answer" = "y" ]; then
  #openssl dgst -sha512 "$volume_path"
  openssl dgst -sha512 "$mount_point/Versioning"
fi

printf "%s\n" "Done"