#date: 2021-12-08T17:16:40Z
#url: https://api.github.com/gists/4f1c385ba57ef2f78eea559729ecc237
#owner: https://api.github.com/users/alexseif

#!/bin/bash

# Check if ran with root permissions
if [ `id -u` -ne 0 ]; then
   printf "The script must be run as root! (you can use sudo)\n"
   exit 1
fi

function arrayContains {
  local e match="$1"
  shift
  for e; do [[ "$e" == "$match" ]] && return 0; done
  return 1
}

function get_available_php_version {
  local BASEDIR="/etc/php/"
  local php_versions_in_etc=$(find $BASEDIR -mindepth 1 -maxdepth 1 -type d -printf '%f ')
  local i=0;
  local available_versions=()
  oldIFS="$IFS"
  for php_version in $php_versions_in_etc
  do
    if $(dpkg --get-selections | grep -v deinstall | grep -q -P "(^php${php_version})"); then
      available_versions[$i]=$php_version
      ((i++))
    fi
  done
  local rtn="$(declare -p available_versions)"
  local IFS=$'\v';
  echo "${rtn#*=}"
  IFS="$oldIFS"
}

function print_available_php_verions {
    eval declare -a avaiable_php_versions="$(get_available_php_version)"
    printf "Basic Usage: bash %s <php-version>\n" "$(basename $0)"
    current_php=$(php --version | head -n 1 | cut -d " " -f 2 | cut  -c-3)
    printf "Current active PHP version is  %s\n" "$current_php"
    printf "Avaliable PHP Versions are:\n"
    for ix in ${!avaiable_php_versions[*]}
    do
      printf "  %s\n" "${avaiable_php_versions[$ix]}"
    done
}

function main {
  if [  -z "$1" ]
  then
    print_available_php_verions
    exit 0
  fi
  local VERSION=$1

  eval declare -a avaiable_php_versions="$(get_available_php_version)"

  if  $(arrayContains "${VERSION}" "${avaiable_php_versions[@]}") ; then
    printf "PHP %s is avaliable in the system.\n" "${VERSION}"
    current_php=$(php --version | head -n 1 | cut -d " " -f 2 | cut  -c-3)
    printf "Switching PHP from %s to %s\n" "$current_php" "$VERSION"
    sudo a2dismod php$current_php
    sudo a2enmod php$VERSION
    sudo service apache2 restart
    sudo update-alternatives --set php /usr/bin/php$VERSION
  else
    print_available_php_verions
  fi
}

main $1