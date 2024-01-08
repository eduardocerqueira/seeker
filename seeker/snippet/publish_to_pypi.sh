#date: 2024-01-08T17:09:26Z
#url: https://api.github.com/gists/7e06761925bf44fc1b570f9bd9ea3481
#owner: https://api.github.com/users/aegilops

#!/bin/bash

python3 -m pip install --upgrade pip
python3 -m pip install --upgrade build
python3 -m pip install --upgrade twine

python3 -m build

twine upload --username __token__ dist/*
