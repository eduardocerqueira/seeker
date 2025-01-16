#date: 2025-01-16T16:38:27Z
#url: https://api.github.com/gists/57f10a3c0e41bf3c1a9b1936602e8672
#owner: https://api.github.com/users/sub314xxl

#!/bin/bash

if [[ ! -d /var/www/app/node_modules ]]; then
  echo "~> installing dependencies"
  yarn install
fi

if [[ ! -f /home/node/bin/node && -f /usr/local/bin/node ]]; then
  echo "~> expose bin"
  cp /usr/local/bin/node /home/node/bin/node
  echo "~> fix permissions"
  chown -R node:node .
fi

echo "Details: '$(pwd)' | '$(quasar -v)'"

echo "~> starting dev"
quasar dev
