#date: 2023-07-04T17:00:49Z
#url: https://api.github.com/gists/c19375bc524d57e88366554f382d2969
#owner: https://api.github.com/users/phuntik

#!/bin/bash

api_url=http://localhost:3001
# TODO jinja template api url?
# TODO nice readme
# TODO auth option

if [ $(curl -o /dev/null -w '%{http_code}\n' -s -I $api_url) != 200 ]
  then
    echo no connection to "$api_url"
    exit 1
  else
    echo connection ok
fi

raw=$(curl -s "$api_url/api/trigger/search?onlyProblems=false&p=0&size=2000" )
echo $raw > raw.json
triggers=$(echo $raw | jq '.list[]')
echo "Processing..."

> delete_requests
for id in $(echo $triggers| jq ".id")
do
  metrics=$(echo $triggers | jq "select(.id=="$id") | (.last_check.metrics | keys[] as \$k | \$k)" | sed 's/ /%20/g')
  for metric in $metrics
  do
    echo $api_url/api/trigger/$id/metrics?name=$metric | sed 's/"//g' >> delete_requests
  done
done

# TODO check mode (if want to check what to be deleted)
for request in $(cat delete_requests)
  do
    [ $(curl -o /dev/null -w '%{http_code}\n' -s -X DELETE "$request") == 200 ] && echo OK "$request"
done

echo DONE
