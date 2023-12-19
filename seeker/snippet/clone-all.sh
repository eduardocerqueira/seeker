#date: 2023-12-19T16:55:50Z
#url: https://api.github.com/gists/a5d0d66a6f01b74224d2cf36207445bb
#owner: https://api.github.com/users/auser

#!/usr/bin/env sh

repos='bridge bridge-react chat cli coreth dao docs exchange explorer faucet finance gsn indexer lattice lpm luxjs marketplace multiparty netrunner netrunner-sdk node oraclevm plugins-core safe safe-ios sites standard subnet-evm town ui vault vmsdk wallet zchain'

for r in $repos;
  do git clone git@github.com:luxdefi/$r
done
