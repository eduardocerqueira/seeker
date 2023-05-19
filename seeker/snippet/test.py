#date: 2023-05-19T17:09:53Z
#url: https://api.github.com/gists/7d7948ecd192d3a98114433811fd099b
#owner: https://api.github.com/users/TinyTinfoil

#!/bin/bash

if [ "$1" == "-h" ]; then
echo -e "Pre-commit test script for python files. \n Input files to test like so: ../../../test.sh file"
exit
fi
coverage run -m pytest *.py
coverage report
black .
pylint *.py
pydocstyle *.py --convention=google --add-ignore=D200,D212,D205,D415