#date: 2022-03-31T17:06:57Z
#url: https://api.github.com/gists/802821d9280d72e8b994fd989403a33b
#owner: https://api.github.com/users/fgbreel

#!/bin/bash

# download installed packages and generate local debian repository
# this command can make your computer very busy! watch out!
dpkg -l | tail -n +5 | awk '{print $2}' | while read line; do (apt download $line -t sid &); done && dpkg-scanpackages ./ > Packages && gzip -k -f Packages