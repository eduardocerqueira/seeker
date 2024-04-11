#date: 2024-04-11T16:49:05Z
#url: https://api.github.com/gists/ae3fe1f68f76483fe8babd902b737b14
#owner: https://api.github.com/users/pythoninthegrass

#!/usr/bin/env bash

# $USER
[[ -n $(logname >/dev/null 2>&1) ]] && logged_in_user=$(logname) || logged_in_user=$(whoami)

# $UID
# logged_in_uid=$(id -u "${logged_in_user}")

# $HOME
logged_in_home=$(eval echo "~${logged_in_user}")

# also symlinked to ~/.local/bin/unlock-vault
export ANSIBLE_VAULT_PASSWORD_FILE= "**********"
export no_proxy='*'
_vault.sh"
export no_proxy='*'
