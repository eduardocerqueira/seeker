#date: 2024-05-31T16:56:16Z
#url: https://api.github.com/gists/1d0a24b93cc6ea05f424049eb9eec261
#owner: https://api.github.com/users/VTLJR

#!/usr/bin/env bash

set -euo pipefail

brew install lima
mkdir -p $HOME/.lima/docker
wget --output-document ~/.lima/docker/lima.yaml https://raw.githubusercontent.com/lima-vm/lima/master/examples/docker.yaml
echo "NoHostAuthenticationForLocalhost yes" >> ~/.ssh/config
