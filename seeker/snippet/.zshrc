#date: 2022-03-28T16:53:53Z
#url: https://api.github.com/gists/1b202a297cebcca5c64382757f44661a
#owner: https://api.github.com/users/angelikatyborska

npm() {
  if [ -f yarn.lock ]; then
    echo 'use yarn';
  else
    command npm $*;
  fi
}

yarn() {
  if [ -f package-lock.json ]; then
    echo 'use npm';
  else
    command yarn $*;
  fi
}