#date: 2023-01-05T16:52:45Z
#url: https://api.github.com/gists/fbf7c4fe6c776107fd7c843e5c956b8c
#owner: https://api.github.com/users/liusanchuan

export NODE_VERSION=16.13.0
apt install -y curl
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
export NVM_DIR=/root/.nvm
. "$NVM_DIR/nvm.sh" && nvm install ${NODE_VERSION}
. "$NVM_DIR/nvm.sh" && nvm use v${NODE_VERSION}
. "$NVM_DIR/nvm.sh" && nvm alias default v${NODE_VERSION}
export PATH="/root/.nvm/versions/node/v${NODE_VERSION}/bin/:${PATH}"
node --version
npm --version

# install yarn
npm install --global yarn