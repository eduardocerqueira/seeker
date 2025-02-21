#date: 2025-02-21T16:56:48Z
#url: https://api.github.com/gists/916781c84c894825924c385fbdc5307b
#owner: https://api.github.com/users/FrodoBaggins52

#!/bin/bash
# This script downloads the latest discord .deb file and saves it to your home folder
# It then installs the .deb file and removes it afterwards

#Change to your username
USERNAME="example"

cd "/home/$USERNAME/"
rm ./temp-discord
wget -O "temp-discord" "https://discord.com/api/download?platform=linux&format=deb"
sudo dpkg -i ./temp-discord
rm ./temp-discord

exit