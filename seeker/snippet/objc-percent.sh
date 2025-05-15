#date: 2025-05-15T17:04:31Z
#url: https://api.github.com/gists/de3e3ac09c5ed171ed968446056f5f0c
#owner: https://api.github.com/users/capricorn

#!/bin/bash

cloc --csv . | awk -F "," '$2 == "Swift" { swift_loc=$5 } $2 == "Objective-C" { c_loc=$5 } END { printf("%.2f", (c_loc/(swift_loc+c_loc)*100)) }'
