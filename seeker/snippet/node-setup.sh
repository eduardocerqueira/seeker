#date: 2023-01-19T16:41:03Z
#url: https://api.github.com/gists/b00bac976f363804c4db259c4421c140
#owner: https://api.github.com/users/aledosreis

#!/bin/sh

echo "Be sure to install NodeJS from http://nodejs.org before continuing"

read -p "Press enter to continue"

# NPM proxy settings
echo "Configuring NodeJS..."
mkdir /c/Users/$USERNAME/npm/
touch /c/Users/$USERNAME/npm/.npmrc
echo "proxy=http://gateway.zscaler.net:80/" >> /c/Users/$USERNAME/npm/.npmrc

# Tell NPM to use a local directory for installations and caching because user profile folders that are mapped to network shares cause many problems
mkdir /c/apps
mkdir /c/apps/npm
mkdir /c/Program\ Files/nodejs/node_modules/npm/
touch /c/Program\ Files/nodejs/node_modules/npm/.npmrc
cp /c/Program\ Files/nodejs/node_modules/npm/.npmrc /c/Program\ Files/nodejs/node_modules/npm/.npmrc.backup
echo "prefix=C:\apps\npm" > /c/Program\ Files/nodejs/node_modules/npm/.npmrc