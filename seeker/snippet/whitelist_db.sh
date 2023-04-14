#date: 2023-04-14T17:01:12Z
#url: https://api.github.com/gists/233290faeefc3e9649b6a2353c158c28
#owner: https://api.github.com/users/filipeandre

#!/usr/bin/env bash

while getopts ":e:r:" opt; do
case "$opt"
in
    e) ENV=$(echo "$OPTARG" | tr '[A-Z]' '[a-z]');;
    r) REGION=$OPTARG;;
esac
done

DB=iownit-system-db-$ENV
echo "DB: ${DB}"
echo "Region: ${REGION}"

SG=$(aws rds describe-db-instances --region ${REGION} --query "DBInstances[? starts_with(DBInstanceIdentifier, '${DB}')].VpcSecurityGroups[0].VpcSecurityGroupId" --output text)
echo "Security group: ${SG}"

CIDR="$(curl -s checkip.amazonaws.com)/32"
echo "My ip: ${CIDR}"

HOST_NAME="$(scutil --get LocalHostName)"
echo "My host name: ${HOST_NAME}"

aws ec2 authorize-security-group-ingress \
    --region ${REGION} \
    --group-id $SG \
    --ip-permissions IpProtocol=tcp,FromPort=5432,ToPort=5432,IpRanges="[{CidrIp=$CIDR,Description="$HOST_NAME"}]"

echo "Added $CIDR to ${SG}"

subnet=$(aws rds describe-db-instances --region ${REGION} --query "DBInstances[? starts_with(DBInstanceIdentifier, '${DB}')].DBSubnetGroup.Subnets[0].SubnetIdentifier" --output text)
echo "Target subnet: ${subnet}"

acl=$(aws ec2 describe-network-acls --region ${REGION} --filters "Name=association.subnet-id,Values=$subnet" --query "NetworkAcls[0].NetworkAclId" --output text)
echo "Target acl: ${acl}"

CIDR="0.0.0.0/0"
aws ec2 create-network-acl-entry --region ${REGION} --cidr-block ${CIDR} --ingress --network-acl-id ${acl} --rule-number 200 --protocol tcp --port-range From=5432,To=5432 --rule-action allow
echo "Added $CIDR to ${acl}"
