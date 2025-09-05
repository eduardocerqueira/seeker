#date: 2025-09-05T16:42:33Z
#url: https://api.github.com/gists/e9673f5ce817016779717315fb4b582b
#owner: https://api.github.com/users/Tiger-Foxx

#!/usr/bin/env bash
set -euo pipefail
apt update && apt install -y build-essential libssl-dev git
cd /opt
git clone https://github.com/wg/wrk.git
cd wrk
make
cp wrk /usr/local/bin/
echo "wrk installed" > /root/wrk_install_done.txt
