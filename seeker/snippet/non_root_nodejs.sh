#date: 2023-01-31T16:58:06Z
#url: https://api.github.com/gists/5643855a7fbcf91527c09f2b3fc1c733
#owner: https://api.github.com/users/ferdousulhaque

wget https://nodejs.org/download/release/v16.19.0/node-v16.19.0-linux-x64.tar.gz
tar xvf node-v16.19.0-linux-x64.tar.gz

mv node-v16.19.0-linux-x64 nodejs
mkdir ~/bin
cp nodejs/bin/node ~/bin
cd ~/bin
ln -s ../nodejs/lib/node_modules/npm/bin/npm-cli.js npm

node --version
npm --version