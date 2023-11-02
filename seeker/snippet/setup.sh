#date: 2023-11-02T16:59:14Z
#url: https://api.github.com/gists/24aa0884f72340aaf4c6ef982325dbe1
#owner: https://api.github.com/users/Muffinman

#!/bin/bash

export TYPESENSE_API_KEY=xyz

mkdir $(pwd)/typesense-data

docker run -p 8108:8108 \
            -v$(pwd)/typesense-data:/data typesense/typesense:0.25.1 \
            --data-dir /data \
            --api-key=$TYPESENSE_API_KEY \
            --enable-cors
