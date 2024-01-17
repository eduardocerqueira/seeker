#date: 2024-01-17T16:43:44Z
#url: https://api.github.com/gists/da2fe6042c864fd8f7c8e9194757832c
#owner: https://api.github.com/users/beastycoding

#!/bin/bash

for bucket in $(radosgw-admin bucket list | jq -r .[]); do
    bucket_id=$(radosgw-admin metadata get bucket:${bucket} | jq -r .data.bucket.bucket_id)
    marker=$(radosgw-admin metadata get bucket:${bucket} | jq -r .data.bucket.marker)
    for instance in $(radosgw-admin metadata list bucket.instance | jq -r .[] | grep "^${bucket}:" | grep -v ${bucket_id} | grep -v ${marker}| cut -f2 -d':'); do
         echo "${bucket}: ${instance}"
         radosgw-admin bi purge --bucket=${bucket} --bucket-id=${instance}
         radosgw-admin metadata rm bucket.instance:${bucket}:${instance}
    done
done
