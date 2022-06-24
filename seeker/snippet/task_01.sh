#date: 2022-06-24T17:02:30Z
#url: https://api.github.com/gists/375897f1df0eb85a5320e84045f83a83
#owner: https://api.github.com/users/syedali3762

#!bin/bash
NOTROOT=100
ROOT_ID=0

if [[ $UID -eq "$ROOT_ID" ]]

then
  echo "This is root user."
  which nginx >/dev/null || apt install nginx -y && apt upgrade nginx -y
  else
  exit $NOTROOT
  fi

