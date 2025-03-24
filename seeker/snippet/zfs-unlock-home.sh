#date: 2025-03-24T16:53:27Z
#url: https://api.github.com/gists/3d6ba4d9e01fd49bfca73cf7b7eaf8d2
#owner: https://api.github.com/users/michaelmrose

#!/bin/bash

# Adjust to match your dataset
ZFS_DATASET="trident/home/michael"

# Only try if dataset is encrypted and not mounted
if ! zfs get -H -o value mounted "$ZFS_DATASET" | grep -q "yes"; then
  echo "$PAM_AUTHTOK" | zfs load-key "$ZFS_DATASET"
  zfs mount "$ZFS_DATASET"
fi