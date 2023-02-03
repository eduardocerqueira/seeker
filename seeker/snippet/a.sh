#date: 2023-02-03T16:58:46Z
#url: https://api.github.com/gists/03aafbebe42bdf194252b6a0e079d4e4
#owner: https://api.github.com/users/inappropriatecontent

#! /bin/bash
sudo apt-get update &&
sudo apt-get upgrade -y &&
sudo apt-get install build-essential -y &&
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.3/install.sh | bash -
exit 1