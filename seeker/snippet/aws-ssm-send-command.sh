#date: 2024-01-22T17:06:12Z
#url: https://api.github.com/gists/a6f9140d88f03aa42425dc7f97258175
#owner: https://api.github.com/users/matsuoka

#!/bin/bash
usage(){
	cat <<__USAGE__
NAME
	Execute shell command line on single EC2 instance via SessionManager.

SYNOPSIS
	$(basename $0) INSTANCE_ID COMMAND [COMMAND]...

DESCRIPTION
	This execute shell command line on specified EC2 instance using
	"aws ssm send-command --document-name 'AWS-RunShellScript'".

	'AWS-RunShellScript' document returns 'CommandId', then "aws ssm list-
	command-invocations" deal with 'Status', 'Output', and others.

	This script throws both commands to execute command line and get results
	with observing 'Status'.

__USAGE__
}
WAIT_TIME=5
if [ $# -lt 2 ];then
	usage
	exit 1
fi

set -eu
INSTANCE_ID=$1
shift
COMMAND_ID=$(\
	aws ssm send-command \
	--instance-ids $INSTANCE_ID \
	--document-name 'AWS-RunShellScript' \
	--timeout-seconds=300 --max-errors=0 \
	--parameters commands="\"$*\"" \
	| jq -r .Command.CommandId)

WAIT=1
while [ $WAIT -eq 1 ];do
	RESULT=$(\
		aws ssm list-command-invocations \
		--command-id $COMMAND_ID --details \
		| jq -r .CommandInvocations[0].CommandPlugins[0])
	STATUS=$(echo $RESULT | jq -r .Status)
	STATUS_DETAILS=$(echo $RESULT | jq -r .StatusDetails)
	if [ "$STATUS" = "InProgress" ];then
		echo "Status: $STATUS ..." >&2
		sleep $WAIT_TIME
	else
		WAIT=0
	fi
done
if [ "$STATUS_DETAILS" == "Success" ];then
	echo $RESULT | jq -r .Output \
		| awk '/^----------ERROR-------$/{ exit 0 }{ print }'
	exit 0
else
	echo $RESULT | jq -r .Output \
		| awk '/^----------ERROR-------$/{err=1}{ if(err){ print > "/dev/stderr" }else{ print } }'
	exit 1
fi
