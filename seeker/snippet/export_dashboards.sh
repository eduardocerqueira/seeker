#date: 2021-09-10T16:52:37Z
#url: https://api.github.com/gists/644c3d9324b1f5059eba9e3983c3c79a
#owner: https://api.github.com/users/ilude

#!/bin/bash

set -o errexit
set -o pipefail

# import .env variables
set -o allexport
source .env
set +o allexport

if [ -z "${GRAFANA_HOST}" ]; then
    echo "GRAFANA_HOST is unset or set to the empty string. Please add GRAFANA_HOST=<hostname> to your .env file"
fi

if [ -z "${GRAFANA_API_KEY}" ]; then
    echo "GRAFANA_API_KEY is unset or set to the empty string. Please add GRAFANA_API_KEY=<api key> to your .env file"
fi

API_URL="https://api_key:$GRAFANA_API_KEY@$GRAFANA_HOST"
set -o nounset

echo "Exporting Grafana dashboards from $GRAFANA_HOST"
rm -rf dashboards
mkdir -p dashboards
for dash in $(curl -s "$API_URL/api/search?query=&" | jq -r '.[] | select(.type == "dash-db") | .uid'); do
        echo "curl -s \"$API_URL/api/dashboards/uid/$dash\" | jq ."
        dashboard_json=$(curl -s "$API_URL/api/dashboards/uid/$dash" | jq .)
        slug=$(echo "$dashboard_json" | jq -r '.meta.slug')
        dashboard_dir=$(echo "$dashboard_json" | jq -r '.meta.folderTitle')
        mkdir -p dashboards/$dashboard_dir
        echo "$dashboard_json" | jq '.dashboard.id = null' | jq '.dashboard' > dashboards/$dashboard_dir/${slug}-${dash}.json
done