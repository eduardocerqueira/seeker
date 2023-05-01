#date: 2023-05-01T17:00:38Z
#url: https://api.github.com/gists/59181d11376be83064535446c749b391
#owner: https://api.github.com/users/redboo

#!/bin/bash

py_init() {
    if [[ ! -d "$(dirname "$1")/$(basename "$1")" ]]; then
        mkdir -p "$(dirname "$1")/$(basename "$1")"
    fi

    cd "$(dirname "$1")/$(basename "$1")"

    python -m venv env
    source ./env/bin/activate
    pip install --upgrade pip
    pip install flake8 black
    cp ~/.config/Code/User/ProjectTemplates/python-gitignore/.gitignore ./
    cp ~/.config/Code/User/ProjectTemplates/GNU_GPL_v3/LICENSE ./
    cp -r ~/.config/Code/User/ProjectTemplates/GITHUB_TEMPLATES/.github ./
    touch requirements.txt main.py
    git init
    code .
}

py_init "$1"
