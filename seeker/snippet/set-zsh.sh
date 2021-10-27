#date: 2021-10-27T17:07:31Z
#url: https://api.github.com/gists/9e69cb9952c51638dacc15f0949a454a
#owner: https://api.github.com/users/bsnux

#!/bin/bash
#
# Set zsh as default shell when auth is done by LDAP
#
set -eou pipefail
getent passwd afernandez | perl -pne 's/bash/zsh/' | sudo tee -a /etc/passwd