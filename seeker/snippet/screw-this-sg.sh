#date: 2023-03-01T16:59:08Z
#url: https://api.github.com/gists/079c2345d8b3a50165e6a2161ba977af
#owner: https://api.github.com/users/sblack4

#!/bin/bash

# abbreviations refer to resource IDs
# security group = SG
# security group rule = SGR
# referencing security group = RGS

# SG to delete
SG=$1

get_SGs_with_rule() {
	aws ec2 describe-security-groups \
		--filters Name=ip-permission.group-id,Values=$SG > groups.json
}

count_groups() {
	echo "Number of groups: $(cat groups.json | jq '.SecurityGroups | length')"
}

get_group_rules() {
	RSG=$1
	aws ec2 describe-security-group-rules \
		--filters Name=group-id,Values=$RSG
}

delete_rule(){
	SGR=$2
	aws ec2 revoke-security-group-ingress \
		--group-id $1 \
		--security-group-rule-ids $SGR
}

iterate_groups() {
	mkdir groups
	for RSG in $(cat groups.json | jq -r '.SecurityGroups[].GroupId'); do
		echo $RSG
		# get the rules
		get_group_rules $RSG > "groups/$RSG.json"
		# get the rule ID
		SGR=$(cat "groups/$RSG.json" | jq -r '.SecurityGroupRules[]|select(.ReferencedGroupInfo.GroupId=="'$SG'").SecurityGroupRuleId')
		#remove it
		echo "deleting $SGR on $RSG"
		delete_rule $RSG $SGR
	done
}

delete_og(){
	aws ec2 delete-security-group --group-id $SG
}

clean_up() {
	rm -rf groups
	rm groups.json
}

# VPC_ID=$(aws ec2 describe-security-groups --group-ids $SG | jq '.SecurityGroups[].VpcId')
get_SGs_with_rule
count_groups
iterate_groups
delete_og
clean_up
