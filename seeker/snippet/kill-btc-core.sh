#date: 2021-09-28T17:08:08Z
#url: https://api.github.com/gists/6c99f94962e24a2d2f809a732347d32c
#owner: https://api.github.com/users/ChristopherA

#!/usr/bin/env bash

# Will find bitcoin-core, bitcoin-qt, etc. and terminate their processes.

# TBD: more graceful shutdown first by sending RPC commands

for i in `ps aux | grep bitcoin | grep -v | awk '{print $2}'` do
  kill $i
done
