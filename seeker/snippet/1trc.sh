#date: 2024-02-19T16:56:11Z
#url: https://api.github.com/gists/c5aa4215c3ffae56476f74515ddb286a
#owner: https://api.github.com/users/gingerwizard

#!/bin/bash
if [[ -z "$CLOUD_ID" || -z "$CLOUD_SECRET" || -z "$AWS_ACCESS_KEY_ID" || -z "$AWS_SECRET_ACCESS_KEY" ]]; then
   echo "Error: Required environment variables are not set."
   exit 1
fi

# identify the organization to create the service in
ORG_ID=$(curl --silent --user $CLOUD_ID: "**********"://api.clickhouse.cloud/v1/organizations | jq -r '.result[0].id')
ORG_NAME=$(curl --silent --user $CLOUD_ID: "**********"://api.clickhouse.cloud/v1/organizations | jq -r '.result[0].name')

echo "using first org '${ORG_NAME}' with id ${ORG_ID} to create service..."
start_time=$(date +%s)
name_suffix=$(xxd -l3 -p /dev/urandom)
service_name="1trc-${name_suffix}"
request="{\"name\": \"${service_name}\", \"provider\": \"aws\", \"region\": \"us-east-1\", \"tier\": \"production\", \"ipAccessList\": [{\"description\": \"Anywhere\", \"source\": \"0.0.0.0/0\" }], \"minTotalMemoryGb\": 720, \"maxTotalMemoryGb\": 720, \"idleScaling\": false}"

# create the service of the specification i.e. 720GB and 180 cores
RESPONSE=$(curl -X POST -H 'Content-Type: "**********":${CLOUD_SECRET} "https://api.clickhouse.cloud/v1/organizations/${ORG_ID}/services" -d "${request}")

SERVICE_ID=$(echo ${RESPONSE} | jq -r '.result.service.id')
HOST=$(echo ${RESPONSE} | jq -r '.result.service.endpoints[] | select(.protocol == "nativesecure") | .host')
PASSWORD= "**********"

# wait for the service to start
echo "service '${service_name}' created with id ${SERVICE_ID}, waiting for service to start..."
sleep 5

while true; do
   state=$(curl --silent --user "$CLOUD_ID: "**********"://api.clickhouse.cloud/v1/organizations/${ORG_ID}/services/${SERVICE_ID}" | jq -r '.result.state')
   if [[ $state == "running" ]]; then
       echo "Service is running"
       break
   else
       echo "Service is not yet running, state is '${state}', waiting for 5 seconds..."
       sleep 5
   fi
done

end_time=$(date +%s)
total_time=$((end_time - start_time))
# report the time to create the service
echo "total time taken to start cluster: $total_time seconds"

# run the query
echo "running query..."
query_time=$(clickhouse client -t --secure --host ${HOST} --password ${PASSWORD} --query "SELECT station, min(measure), max(measure), round(avg(measure), 2) FROM s3Cluster('default','https: "**********"
echo "query took ${query_time}s"

# stop the service
echo "stopping service..."
curl -X PATCH  -H 'Content-Type: "**********":$CLOUD_SECRET "https://api.clickhouse.cloud/v1/organizations/${ORG_ID}/services/${SERVICE_ID}/state" -d '{"command": "stop"}'

# wait for the service to stop
while true; do
   state=$(curl --silent --user "$CLOUD_ID: "**********"://api.clickhouse.cloud/v1/organizations/${ORG_ID}/services/${SERVICE_ID}" | jq -r '.result.state')
   if [[ $state == "stopped" ]]; then
       echo "Service is stopped"
       break
   else
       echo "Service is not yet stopped, state is '${state}', waiting for 5 seconds..."
       sleep 5
   fi
done

echo "done": "**********"://api.clickhouse.cloud/v1/organizations/${ORG_ID}/services/${SERVICE_ID}" | jq -r '.result.state')
   if [[ $state == "stopped" ]]; then
       echo "Service is stopped"
       break
   else
       echo "Service is not yet stopped, state is '${state}', waiting for 5 seconds..."
       sleep 5
   fi
done

echo "done"o "done"