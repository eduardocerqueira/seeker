#date: 2022-01-31T17:07:47Z
#url: https://api.github.com/gists/1385e228c86a2b8e3cb1600bff350dad
#owner: https://api.github.com/users/olg200492

#!/usr/bin/env bash

for f in *.com; do
    mv -- "$f" "${f%.com}.txt"
done