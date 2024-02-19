#date: 2024-02-19T17:04:08Z
#url: https://api.github.com/gists/518c02c1007e1b4f482a912b0eccfc35
#owner: https://api.github.com/users/0xabu

#!/bin/bash

# Construct a python3 venv in TrueNAS

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 VENV_DIR" > /dev/stderr
  exit 1
fi

set -ex

python3 -m venv $1 --without-pip
curl -fSL https://bootstrap.pypa.io/get-pip.py | $1/bin/python3
