#date: 2023-04-27T17:07:37Z
#url: https://api.github.com/gists/3ca630eca73cc55d4461ed8e88e2decf
#owner: https://api.github.com/users/shollingsworth

#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

trap "exit" INT TERM ERR
trap "kill 0" EXIT

aws logs describe-log-groups \
    --log-group-name-prefix "/aws/lambda/sh-ws-demo-backend" \
    --query 'logGroups[].logGroupName' \
    | jq -r '.[]' | while read logGroupName; do
        echo "Tailing $logGroupName"
        aws logs tail \
            "$logGroupName" \
            --since 10m \
            --follow &
    done

while true; do
    sleep 1
done
