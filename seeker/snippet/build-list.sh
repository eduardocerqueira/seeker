#date: 2022-09-16T21:59:25Z
#url: https://api.github.com/gists/397c28dcd387bd465f346c0ee6063039
#owner: https://api.github.com/users/tiny-tinker

#!/usr/bin/env bash

if [ -z "$1" ]
  then
    echo "Usage: build-list.sh BILLING-ACCOUNT-HERE"
    exit
fi

BILLING_ACCT=$1
echo "Using billing acct: " $BILLING_ACCT
echo "PROJECT_NAME,INSTANCE_NAME,ZONE,MACHINE-TYPE,OPERATING_SYSTEM,CPU,MEMORY,DISK1_SIZE,DISK2_SIZE,DISK3_SIZE,DISK4_SIZE,DISK5_SIZE,DISK6_SIZE,DISK7_SIZE,DISK8_SIZE,DISK9_SIZE,DISK10_SIZE" > compute-engine-details.csv
prjs=( $( gcloud projects list --format="value(PROJECT_ID)" --billing-account=$BILLING_ACCT) )
for i in "${prjs[@]}"
do
    echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
    echo "Setting Project: $i"

    # Check if compute is enabled
    compute=$( gcloud services list --enabled --project=$i | grep -c compute.googleapis.com )
    if [ $compute -eq 0 ]; then
        echo "... Compute not enabled, continuing"
        continue
    fi
    gcloud compute instances list --project=$i --format="value(NAME,ZONE)" | while read line; do echo "$i $line"; done | xargs -n3 sh -c 'python3 retrieve-compute-engine-details.py $1 $2 $3 >> compute-engine-details.csv' sh
done
