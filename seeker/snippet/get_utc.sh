#date: 2023-05-01T16:56:14Z
#url: https://api.github.com/gists/1293dd2c3654d49200b3ae2464c5a2f5
#owner: https://api.github.com/users/thesuhu

#!/bin/bash

# get UTC date
date -u -d "$(curl -sI google.com | grep -i '^date:' | cut -d' ' -f2-)"

# get local date from UTC
date -d "$(curl -sI google.com| grep -i '^date:'|cut -d' ' -f2-)"

# sync local date from UTC
sudo date -s "$(date -d "$(curl -sI google.com | grep -i '^date:' | cut -d' ' -f2-)" '+%Y-%m-%d %H:%M:%S')"

# UTC vs local
 date -d "$(curl -sI google.com| grep -i '^date:'|cut -d' ' -f2-)" && date
 date -u -d "$(curl -sI google.com| grep -i '^date:'|cut -d' ' -f2-)" && date -u