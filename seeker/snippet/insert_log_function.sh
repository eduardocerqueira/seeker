#date: 2023-05-29T16:42:07Z
#url: https://api.github.com/gists/72059fbde139ee39d66cc401942f6685
#owner: https://api.github.com/users/ryanc410

#!/usr/bin/env bash

d=$(date +%D)
t=$(date +%T)
timestamp="${t}"["${d}"]-
logfile=script.log

function insertlog(){
  logmessage="$1"
  echo "${timestamp}"-"${logmessage}" >> "${logfile}"
}

# USAGE
#=========================#
if [[ $EUID -ne 0 ]]; then
  insertlog "ERROR: Script executed without using sudo"
fi

