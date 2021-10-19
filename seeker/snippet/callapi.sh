#date: 2021-10-19T17:07:19Z
#url: https://api.github.com/gists/5fdbd863840ec1f3fcee52d72fcd4342
#owner: https://api.github.com/users/lalanikarim

#!/bin/bash

verbose=''
while getopts vm:e:d:q:s: flag
do
    case "${flag}" in
        m) verb=${OPTARG};;
        e) endpoint=${OPTARG};;
        d) body="-d '${OPTARG}'";;
        q) query=" | jq -r '${OPTARG}'";;
        v) verbose="true";;
        s) store=${OPTARG};;
    esac
done

if [ -z "${verb}" ] 
then
    verb=GET
fi

if [ -z "${endpoint}" ]
then
    endpoint=/
fi

if [ -z "${API}" ]
then
    echo "API not set"
else
    accept_header="-H 'accept: application/json'"

    if [ -z "${TOKEN}" ]
    then
    #    echo "TOKEN not set. Sending request without authorization header"
    :
    else
        authorization_header="-H 'authorization:Bearer $TOKEN'"
    fi

    # echo "Verb: $verb";
    # echo "EndPoint: $endpoint";
    # echo "Body: $body";

    command="curl -X $verb $API$endpoint $accept_header $authorization_header $body"

    if [ -z "${verbose}" ]
    then
        :
    else
        echo "Request:"
        echo
        echo $command $query
        echo
        echo "Response:"
        echo
    fi
    if [ -z "${query}" ]
    then
        eval $command
    else
        if [ -z "${store}" ]
        then
            eval "$command $query"
        else ### This bit doesn't work Not sure if it even is supposed to but adding it here for reference
            echo "export $store=\$($command $query)"
            eval "export $store=\$($command $query)"
        fi
    fi
fi