#date: 2024-02-14T16:54:21Z
#url: https://api.github.com/gists/33435cbef0b6322e8584e3fcc1123240
#owner: https://api.github.com/users/manelatun

#!/bin/bash

# Install Node Version Manager.
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/master/install.sh | bash
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"

# Install latest Node.js and Yarn.
nvm install --lts
nvm use --lts
npm install -g npm@latest
npm install -g yarn@latest

echo 'Restart your shell to use Nodejs.'
