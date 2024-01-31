#date: 2024-01-31T16:51:34Z
#url: https://api.github.com/gists/2601784e854045e0c5949e69c63d624a
#owner: https://api.github.com/users/marcolussetti

#!/bin/bash
# Takes a set of haproxy logs, and parses them in GoAccess
# By Marco Lussetti
# Some caveats:
# 1. This uses GoAccess in a container
# 2. HAProxy works off frontends and backends, not the usual format GoAccess expects, so:
# 2a. Frontends are logged as "Remote User (HTTP authentication)"
# 2b. Backends are logged as "Virtual Hosts"

zcat --force haproxy.log* | docker run --rm -i -e LANG=$LANG allinurl/goaccess -a -o html --log-format='%^]%^ %h:%^ [%d:%t.%^] %e %v/%R/%^/%^/%L/%^ %s %b %^"%r"' --date-format='%d/%b/%Y' --time-format='%H:%M:%S' - > report.html