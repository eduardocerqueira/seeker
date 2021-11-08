#date: 2021-11-08T16:59:03Z
#url: https://api.github.com/gists/43e2e3ac6e3744f81f4f7822dd8fc077
#owner: https://api.github.com/users/mrchapp

#!/bin/bash

curl -sSL https://lkft.validation.linaro.org/scheduler/device_types | sed -e 's#<nav #<!--nav #g' -e 's#</nav>#</nav-->#g' > device_types

#for board in dragonboard-410c dragonboard-845c; do
for board in $(xmllint -format -html --xpath "//table/tbody/tr/td[1]/a/text()" device_types); do
    echo -n "${board},"
    for status in maintenance offline; do
        xp="//table/tbody/tr[td[1]/a/text() = \"${board}\"]/td[@class=\"${status}\"]/text()";
        xmllint -format -html --xpath "${xp}" device_types 2>&1 | sed -e 's#XPath set is empty#0#g'
    done | awk '{total += $1} END {print total}'
done

rm device_types