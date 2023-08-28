#date: 2023-08-28T16:54:12Z
#url: https://api.github.com/gists/e2ae7453cb25797e478c04b2b3b93214
#owner: https://api.github.com/users/nemoDreamer

#!/usr/bin/env bash

NAMES=(Black Red Green Yellow Blue Magenta Cyan White)
CHAR="██  "

for i in {0..7}; do
    CODE=$((i + 30))
    BRIGHT=$((i + 90))

    echo -n -e "${NAMES[$i]}\t"
    # blocks
    # - bright
    echo -n -e "\033[${BRIGHT}m${CHAR}"
    # - normal
    echo -n -e "\033[${CODE}m${CHAR}"
    # - faded
    echo -n -e "\033[$CODE;2m${CHAR}"
    # reset
    echo -e "\033[0m\n"
done
