#date: 2025-09-24T17:08:04Z
#url: https://api.github.com/gists/2b9f4229bf08f9cbf52f083d83683564
#owner: https://api.github.com/users/Wonno

#!/usr/bin/env bash

# Create Stack in Portainer using API
#
# - compose.yml
# - stack.env

set -euo pipefail

which curl || (echo "Missing 'curl'"; exit 1)
which jq || (echo "Missing 'jq'"; exit 1)
which jo || (echo "Missing 'jo'"; exit 1)

# read values from config file
if [[ -e portainer.config ]] ; then
  source portainer.config
fi
if [[ -z "${PORTAINER_USER-}" ]]; then
  read -rp "Portainer-User: " PORTAINER_USER
  echo
fi
if [[ -z "${PORTAINER_PASSWORD-}" ]]; then
  read -srp "Portainer-Password: "**********"
  echo
fi
if [[ -z "${PORTAINER_URL-}" ]]; then
  read -rp "Portainer-URL: " PORTAINER_URL
  echo
fi
if [[ -z "${PORTAINER_ENDPOINTNAME-}" ]]; then
  read -rp "Endpoint-Name: " PORTAINER_ENDPOINTNAME
  echo
fi
if [[ -z "${PORTAINER_STACKNAME-}" ]]; then
  read -rp "Stack-Name: " PORTAINER_STACKNAME
  echo
fi

# Get Bearer Token
PORTAINER_TOKEN= "**********"
    --request POST "${PORTAINER_URL}/api/auth" \
    --data "$(jo username= "**********"="${PORTAINER_PASSWORD}")" \
| jq -r '.jwt')

# Fetch ID by name
PORTAINER_ENDPOINTID=$(curl --silent --fail --show-error \
    --request GET "${PORTAINER_URL}/api/endpoints" \
    --header "Authorization: "**********"
| jq --arg name "${PORTAINER_ENDPOINTNAME}" '.[] | select (.Name=="$name").Id ')

declare -A properties
# Read File "stack.env" into  associative array; skip empty lines; skip commented lines
while IFS='=' read -r key value; do
    properties["$key"]="$value"
done < <(
  # shellcheck disable=SC2002
  cat stack.env \
    | tr -s '\n' \
    | grep -Ev "^[[:space:]]*#.*"
  )

# Convert associative array to JSON array with name value pairs
STACKENV=$(for key in "${!properties[@]}"; do
    jo -- -s name="${key}" -s value="${properties[$key]}"
done | jo -a -p)

# Create the Portainer stack
curl --silent --fail --show-error \
    --request POST "${PORTAINER_URL}/api/stacks/create/standalone/file" \
    --header "Authorization: "**********"
    --data "Name=${PORTAINER_STACKNAME}" \
    --data "endpointId=${PORTAINER_ENDPOINTID}" \
    --data-urlencode "Env=${STACKENV}" \
    --data @compose.yaml
DPOINTID}" \
    --data-urlencode "Env=${STACKENV}" \
    --data @compose.yaml
