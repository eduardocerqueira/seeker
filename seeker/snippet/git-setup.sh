#date: 2023-01-19T16:41:03Z
#url: https://api.github.com/gists/b00bac976f363804c4db259c4421c140
#owner: https://api.github.com/users/aledosreis

#!/bin/sh

# Git proxy settings
echo "Configuring Git for compatibility with ZScaler..."
git config --global http.proxy http://gateway.zscaler.net:80/
git config --system http.proxy http://gateway.zscaler.net:80/