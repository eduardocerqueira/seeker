#date: 2023-04-13T16:49:07Z
#url: https://api.github.com/gists/8e3101037f67f1919e6ee1d5f0113317
#owner: https://api.github.com/users/andreztz

#!/bin/bash
set -o nounset -o pipefail -o errexit

# Load all variables from .env and export them all for Ansible to read
set -o allexport
source "$(dirname "$0")/.env"
set +o allexport

# Run Ansible
exec ansible-playbook "$@"