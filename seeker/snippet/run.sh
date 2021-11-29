#date: 2021-11-29T17:02:07Z
#url: https://api.github.com/gists/85b5153e0c18b1e086d08d6fc894ca89
#owner: https://api.github.com/users/jac18281828

#!/usr/bin/env bash

VERSION=$(date +%m%d%y)

API_KEY_FILE=apikey.key
if [ -z "${BN_API_KEY}" ]
then
        if [ -f "${API_KEY_FILE}" ]
        then
           API_KEY=$(cat ${API_KEY_FILE})
        else
           echo "${API_KEY_FILE} not found"
           exit 1
       fi
else
        API_KEY=${BN_API_KEY}
fi

docker build . -t subscribe:${VERSION} && \
     docker run --name subscribe -e BN_API_KEY=${API_KEY} --rm -i -t subscribe:${VERSION} 
