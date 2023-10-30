#date: 2023-10-30T17:05:55Z
#url: https://api.github.com/gists/2ebb1999cae9d6063b24467b5500c646
#owner: https://api.github.com/users/doodaz

# Simple script to list all RDS instances and then disable Automatic Minor Version Upgrades 

# list RDS instances to a file
aws rds --region eu-west-1 describe-db-instances | jq -r '.DBInstances[].DBInstanceIdentifier' > rds.log

# loop over the items in the file, disable auto minor version upgrades
while read rds; do
    aws rds modify-db-instance --region eu-west-1 --db-instance-identifier ${rds} --no-auto-minor-version-upgrade
    sleep 5
done <rds.log