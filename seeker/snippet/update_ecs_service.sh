#date: 2023-05-29T17:00:54Z
#url: https://api.github.com/gists/1c69f75d20e1765e56b80d1b21838ebc
#owner: https://api.github.com/users/filipeandre

#!/bin/bash

set -e

# possible -b (base / app name) -i (image version), -e (deploy env) and -s (service id)
while getopts i:s:c: option
do
case "${option}"
in
  i) IMG_VERSION=${OPTARG};;
  s) SERVICE_ID=${OPTARG};;
  c) CLUSTER_NAME=${OPTARG};;
esac
done

echo "IMG_VERSION: " $IMG_VERSION
echo "SERVICE_ID: " $SERVICE_ID

if [ -z "$SERVICE_ID" ]; then
    echo "exit: No SERVICE_ID specified"
    exit;
fi

if [ -z "$IMG_VERSION" ]; then
    echo "exit: No IMG_VERSION specified"
    exit;
fi

if [ -z "$CLUSTER_NAME" ]; then
    echo "exit: No CLUSTER_NAME specified"
    exit;
fi

# Define variables
TASK_FAMILY=${SERVICE_ID}
SERVICE_NAME=${SERVICE_ID}

IMAGE_PACEHOLDER="<IMAGE_VERSION>"

CONTAINER_DEFINITION_FILE=$(cat $SERVICE_ID.container-definition.json)
CONTAINER_DEFINITION="${CONTAINER_DEFINITION_FILE//$IMAGE_PACEHOLDER/$IMG_VERSION}"


# Get the existing task definition
task_def_response=$(aws ecs describe-task-definition --task-definition "${TASK_FAMILY}")
task_def_arn=$(echo "$task_def_response" | jq -r '.taskDefinition.taskDefinitionArn')
task_def_role_arn=$(echo "$task_def_response" | jq -r '.taskDefinition.executionRoleArn')


export TASK_VERSION=$(aws ecs register-task-definition \
                        --family ${TASK_FAMILY} \
                        --container-definitions "$CONTAINER_DEFINITION" \
                        --task-role-arn "$task_def_role_arn" | jq --raw-output '.taskDefinition.revision')



echo "Registered ECS Task Definition: " $TASK_VERSION

if [ -n "$TASK_VERSION" ]; then
    echo "Update ECS Cluster: " $CLUSTER_NAME
    echo "Service: " $SERVICE_NAME
    echo "Task Definition: " $TASK_FAMILY:$TASK_VERSION
    DEPLOYED_SERVICE=$(aws ecs update-service --cluster $CLUSTER_NAME --service $SERVICE_NAME --task-definition $TASK_FAMILY:$TASK_VERSION | jq --raw-output '.service.serviceName')
    echo "Deployment of $DEPLOYED_SERVICE complete"

else
    echo "exit: No task definition"
    exit;
fi