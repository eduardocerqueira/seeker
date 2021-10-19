#date: 2021-10-19T17:00:47Z
#url: https://api.github.com/gists/fc63acd691224fa7ad221411af1455fc
#owner: https://api.github.com/users/mridang

#!/bin/bash
aws emr describe-cluster --cluster-id @1 | jq --raw-output '.Cluster .MasterPublicDnsName | sub("ip-(?<ip1>[0-9]*)-(?<ip2>[0-9]*)-(?<ip3>[0-9]*)-(?<ip4>[0-9]*).*$"; "\(.ip1).\(.ip2).\(.ip3).\(.ip4)")'