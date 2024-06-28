#date: 2024-06-28T16:58:45Z
#url: https://api.github.com/gists/8a518063098334f27dd23c41bb44c52e
#owner: https://api.github.com/users/sombriks

#!/bin/sh
# need to run this to have a proper local cluster
sleep 5 ; echo yes | redis-cli --cluster create redis-7003:7003 redis-7001:7001 redis-7002:7002 --cluster-replicas 0
