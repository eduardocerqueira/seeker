#date: 2022-05-31T16:57:50Z
#url: https://api.github.com/gists/30c63e85e59e904f6d422eab2f5ba8f7
#owner: https://api.github.com/users/farukcankaya


#!/bin/bash

# Usage
# Make install_kafka.sh and check_consumers.sh executable via `chmod +x _file_`.
# Then, run the script:
# ./check_consumers.sh -g 'consumer.group.retry' -b 'localhost:9092,localhost:9093'

while getopts g:b: flag
do
    case "${flag}" in
        g) consumer_group_name_to_check=${OPTARG};;
        b) brokers=${OPTARG};;
    esac
done

echo "Check if $consumer_group_name_to_check exists in:"
for broker in ${brokers//,/ }
do
    echo -n "- $broker... "
    consumer_group=`bin/kafka-consumer-groups.sh --list --bootstrap-server $broker | grep $consumer_group_name_to_check`
    #[ -z "$consumer_group" ] && echo "NOT EXIST" || echo "OK"
    if [ -z "$consumer_group" ]
    then
        echo "NOT EXIST"
    else
        echo -n "OK "
        consumer_count=`bin/kafka-consumer-groups.sh --describe --group "$consumer_group" --bootstrap-server $broker --members | wc -l`
        default_count=2
        consumer_count=$((consumer_count-default_count))
        if [ "$consumer_count" -gt "0" ]
        then
            echo "There are $consumer_count consumers"
        else
            echo "CIS"
        fi
    fi
done