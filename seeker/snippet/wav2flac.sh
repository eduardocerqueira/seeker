#date: 2023-10-30T17:04:32Z
#url: https://api.github.com/gists/2ee3c493b07ab40d7b9c5f0ef21e177a
#owner: https://api.github.com/users/musicologyman

#!/bin/bash

SAVEIFS=$IFS
IFS=$(echo -en "\n\b")
for f in *.wav
do
  ffmpeg -i "$f" "${f%%.wav}.flac"
done
IFS=$SAVEIFS
