#date: 2021-10-19T16:59:56Z
#url: https://api.github.com/gists/fce4c7f07a29281ac35df53466a0c9b6
#owner: https://api.github.com/users/Huevos-y-Bacon

#!/usr/bin/env bash
# shellcheck disable=SC2034

# API DOCUMENTATION: https://docs.datadoghq.com/api/latest/aws-integration/

# ============================================================
# TO BE REMOVED AND PROVIDED VIA EXPORTS
DD_API_KEY="DATADOG_API_KEY"
DD_APPLICATION_KEY="DATADOG_APPLICATION_KEY"
ACCOUNTID="AWS_ACCOUNT_ID"
ACCOUNT_NAME="AWS_ACCOUNT_ALIAS"
# ============================================================

ROLE="DatadogAWSIntegrationRole"
DD_SITE=eu

[[ "$*" == *--debug* ]] && set -x
[[ "$*" == *-vvv* ]] && set -x

JSON_PAYLOAD=$(jq -n -r \
  --arg account_id    $ACCOUNTID \
  --arg role_name     $ROLE \
  --arg AccountName   $ACCOUNT_NAME \
  '{ account_id: $account_id, role_name: $role_name, host_tags: [ $AccountName ] }'
)

list_integrations(){
  # LIST INTEGRATIONS
  curl -s -X GET "https://api.datadoghq.${DD_SITE}/api/v1/integration/aws" \
  -H "Content-Type: application/json" \
  -H "DD_API_KEY: ${DD_API_KEY}" \
  -H "DD_APPLICATION_KEY: ${DD_APPLICATION_KEY}" | jq
}

validate_api_key(){
  # VALIDATE API KEY
  curl -s -X GET "https://api.datadoghq.${DD_SITE}/api/v1/validate" \
  -H "Content-Type: application/json" \
  -H "DD-API-KEY: ${DD_API_KEY}" | jq
}

generate_new_external_id(){
  # GENERATE NEW EXTERNAL ID
  ACCOUNT_NAME="AccountName:${ACCOUNT_NAME}"
  curl -s -X PUT "https://api.datadoghq.${DD_SITE}/api/v1/integration/aws/generate_new_external_id" \
  -H "Content-Type: application/json" \
  -H "DD-API-KEY: ${DD_API_KEY}" \
  -H "DD-APPLICATION-KEY: ${DD_APPLICATION_KEY}" \
  -d @- << EOF | jq
$JSON_PAYLOAD
EOF
}

delete_integration(){
  # DELETE AN INTEGRATION - DELETE query
  ACCOUNT_NAME="AccountName:${ACCOUNT_NAME}"
  curl -s -X DELETE "https://api.datadoghq.${DD_SITE}/api/v1/integration/aws" \
  -H "Content-Type: application/json" \
  -H "DD_API_KEY: ${DD_API_KEY}" \
  -H "DD_APPLICATION_KEY: ${DD_APPLICATION_KEY}" \
  -d @- << EOF | jq
$JSON_PAYLOAD
EOF
}

create_integration(){
  # CREATE NEW INTEGRATION - POST query
  ACCOUNT_NAME="AccountName:${ACCOUNT_NAME}"
  echo "JSON_PAYLOAD: $JSON_PAYLOAD"
  curl -s -X POST "https://api.datadoghq.${DD_SITE}/api/v1/integration/aws" \
  -H "Content-Type: application/json" \
  -H "DD-API-KEY: ${DD_API_KEY}" \
  -H "DD-APPLICATION-KEY: ${DD_APPLICATION_KEY}" \
  -d @- << EOF | jq
$JSON_PAYLOAD
EOF
}

update_integration(){
  # UPDATE EXISTING INTEGRATION - PUT query
  ACCOUNT_NAME="AccountName:${ACCOUNT_NAME}"
  echo "JSON_PAYLOAD: $JSON_PAYLOAD"
  curl -s -X PUT "https://api.datadoghq.${DD_SITE}/api/v1/integration/aws" \
  -H "Content-Type: application/json" \
  -H "DD-API-KEY: ${DD_API_KEY}" \
  -H "DD-APPLICATION-KEY: ${DD_APPLICATION_KEY}" \
  -d @- << EOF | jq
$JSON_PAYLOAD
EOF
}

commands(){
  DD_SITE=$(echo ${DD_SITE} | tr '[:lower:]' '[:upper:]')
  echo "
Available commands to interact with the DataDog ${DD_SITE} API (AWS):

  list                -  list_integrations
  validate            -  validate_api_key
  generate            -  generate_new_external_id
  delete_integration  -  delete_integration
  create_integration  -  create_integration
  update_integration  -  update_integration

  [ --debug OR -vvv ] -  verbose output (bash -x)
"
  exit 1
}

# ============================================================
if   [[ -z $1 ]]; then echo "No command specified"; commands
elif [[ "$1" == "list" ]]; then list_integrations
elif [[ "$1" == "validate" ]]; then validate_api_key
elif [[ "$1" == "generate" ]]; then generate_new_external_id
elif [[ "$1" == "delete_integration" ]]; then delete_integration
elif [[ "$1" == "create_integration" ]]; then create_integration
elif [[ "$1" == "update_integration" ]]; then update_integration
else echo "Invalid command"; commands
fi
