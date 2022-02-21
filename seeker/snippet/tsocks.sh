#date: 2022-02-21T17:12:00Z
#url: https://api.github.com/gists/a00286b65a695741cb10146dc54fa201
#owner: https://api.github.com/users/LucHermitte

#!/bin/bash
# Author:       Luc Hermitte <luc.hermitte@gmail.com>
# Purpose:      Script that runs a tsocks in a conf where the passowrd isn't hardcoded
# Licence:      GPLv3
# Version:      1.0.0
# Copyright 2020-2022
#
# Policy
# 1. Check in $http_proxy if the password is present -- TDB
# 2. Check in a gpg encrypted file
#    -> ~/.config/.tsocks-pwd.gpg
# 3. Ask the end user on the fly

_config_home="${XDG_CONFIG_HOME:-${HOME}/.config}"

_tsocks_file="${_config_home}/.tsocks-pwd.gpg"
if [ -f "${_tsocks_file}" ] ; then
    TSOCKS_PASSWORD=$(gpg -q -d "${_tsocks_file}")
fi
if [ ! -v TSOCKS_PASSWORD ] ; then
    read -s -p "Proxy password? " TSOCKS_PASSWORD || exit 127
fi

# echo "pwd: ${TSOCKS_PASSWORD}"
TSOCKS_PASSWORD="${TSOCKS_PASSWORD}" tsocks "$@"
