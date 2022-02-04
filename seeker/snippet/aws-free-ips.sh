#date: 2022-02-04T16:53:54Z
#url: https://api.github.com/gists/fd6c3bcd94f8e246b6f5063405d95057
#owner: https://api.github.com/users/philpennock

#!/bin/sh -eu

: "${VPC_ID:=}"
ShowVPCPrefix=''

format_subnets() {
  jq -r '.Subnets[]
     | "'"${ShowVPCPrefix}"'\(.AvailabilityZone)\t\(.SubnetId)\t\(.CidrBlock)\t\(.AvailableIpAddressCount)\t\(
     if .Tags then .Tags[]|select(.Key=="Name")|.Value else "<none>" end)"' |
    sort
}

if [ ".$VPC_ID" = ".all" ]; then
  VPC_ID=''
  ShowVPCPrefix='\(.VpcId)\t'
fi

if [ ".$VPC_ID" != "." ]; then
  get_subnets() { aws ec2 describe-subnets --filters "Name=vpc-id,Values=[${VPC_ID}]"; }
else
  get_subnets() { aws ec2 describe-subnets; }
fi

get_subnets | format_subnets