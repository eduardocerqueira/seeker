#date: 2023-05-02T16:44:24Z
#url: https://api.github.com/gists/66301ef14d85352dc004769b81c52b03
#owner: https://api.github.com/users/vchrombie

#!/bin/bash

echo "Stage 1"

set -o errexit
set -o pipefail
set -o nounset

echo "Stage 2"

GRAFANA_URL="$1"
headers="Authorization: Bearer $2"
GIT_SRC="$3"
in_path="$GIT_SRC/dashboards_raw"

echo "Stage 3"

echo "Exporting Grafana dashboards from $GRAFANA_URL"
mkdir -p $in_path

echo "Stage 4"

curl -H "$headers" -s "$GRAFANA_URL/api/search?query=&" > tmp.json

echo "Stage 5"

for dash in $(curl -H "$headers" -s "$GRAFANA_URL/api/search?query=&" | jq -r '.[] | select(.type == "dash-db") | .uid'); do
    dash_path="$in_path/$dash.json"

    echo "Stage 6"

    curl -H "$headers" -s "$GRAFANA_URL/api/dashboards/uid/$dash" | jq -r . > $dash_path
    jq -r .dashboard $dash_path > $in_path/dashboard.json
    title=$(jq -r .dashboard.title $dash_path | sed "s/\//-/g")
    folder="$(jq -r '.meta.folderTitle' $dash_path | sed "s/\//-/g")"

    echo "Stage 7"

    mkdir -p "$GIT_SRC/$folder"
    mv -f $in_path/dashboard.json "$GIT_SRC/$folder/${title}.json"
    echo "exported $GIT_SRC/$folder/${title}.json"

    echo "Stage 8"

done

echo "Stage 9"

rm -r $in_path

echo "Stage 10"

