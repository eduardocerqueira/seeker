#date: 2025-11-12T16:43:07Z
#url: https://api.github.com/gists/a821e5a85feec783623e720cdf85ad9b
#owner: https://api.github.com/users/santosh-gouda

#!/bin/bash

if [[ ! -d .git ]]; then
    echo "[*] error: Not a git repo !!!"
    exit 1
fi

precommit_config="https://gist.githubusercontent.com/dhruvSHA256/ae1c759688baee09e2ce60757c4c48eb/raw/e30ce2a600790457ea147ed990b63f434a39de63/.pre-commit-config.yaml"
pylint_config="https://gist.githubusercontent.com/dhruvSHA256/ae1c759688baee09e2ce60757c4c48eb/raw/e30ce2a600790457ea147ed990b63f434a39de63/.pylintrc"
venv="./.venv"

if [[ ! -d ./.venv && ! -d ./venv ]]; then
    echo "[*] No venv found, making: "
    /usr/bin/python3 -m venv .venv
fi

if [[ -z $VIRTUAL_ENV ]]; then
    echo "[*] venv not activated, activating: "
    source $venv/bin/activate
fi

if [[ ! -f "./.pre-commit-config.yaml" ]]; then
    echo "[*] No precommit config found, downloading default precommit config: "
    wget $precommit_config
else
    echo "[*] Existing precommit config found, renaming it "
    mv ".pre-commit-config.yaml" ".pre-commit-config.yaml.bkp"
    wget $precommit_config
fi

if [[ ! -f "./.pylintrc" ]]; then
    echo "[*] No pylint config found, downloading default pylint config: "
    wget $pylint_config
else
    echo "[*] Existing pyling config found, renaming it "
    mv ".pylintrc" ".pylintrc.bkp"
    wget $pylint_config
fi

echo ""
echo -e "[*] Installing required pip modules"
pip install pylint black[d] pre-commit

echo ""
echo "[*] Installing git pre-commit hook in current repo"
pre-commit install

echo ""
echo "[*] running pre-commit hook once for testing"
pre-commit run --all-files