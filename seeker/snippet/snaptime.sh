#date: 2024-11-22T16:53:15Z
#url: https://api.github.com/gists/ce1690b21ae4e587901011a42b1155df
#owner: https://api.github.com/users/oxagast

#!/bin/bash

# these two are user definable vars
LEAVEN=3             # the number of snapshots trailing the one you created that aren't deleted
DO=7                 # needs to be at least 7 days old to be removeable
PTN=("/" "/home");   # subpartitions to snapshot


D=$(date +snap-%d-%m-%Y)
for BTRFSP in "${PTN[@]}"; do
  find "$BTRFSP/snapshots/"  -maxdepth 0 -mtime +$DO -exec ls -1t {} \; | tac | head -n -$LEAVEN | xargs -d '\n' rm -rf --
  if [ ! -d "$BTRFSP/snapshots/$D" ]; then
    sudo btrfs subvolume snapshot $BTRFSP "$BTRFSP/snapshots/$D" && echo "Subvolume snapshot taken: $BTRFSP."
    sudo chmod a+rx,g+rx,u=rwx,o-w "$BTRFSP/snapshots/$D"  && echo "Permission earliest level fixed (a+rx,g+rx,u=rwx,o-w)."
  else
    echo "Already snapped today."
    exit 1
  fi;
done
echo "All snapshots available have been taken."