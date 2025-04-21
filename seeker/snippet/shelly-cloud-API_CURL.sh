#date: 2025-04-21T16:45:00Z
#url: https://api.github.com/gists/53239c480d8b520de5cd33abf7de3898
#owner: https://api.github.com/users/andrekis

#!/bin/bash
SHELLY_KEY=<shelly-key from https://home.shelly.cloud/index.html#/user_settings>
SHELLY_CLOUD_SERVER=<shelly-cloud-server from https://home.shelly.cloud/index.html#/user_settings eg. https://shelly-61-eu.shelly.cloud>
DEVICE_ID= <device-id from  https://home.shelly.cloud>

CSV_OUTPUT_FILE=shelly_values.csv

curl -s -X POST $SHELLY_CLOUD/device/status -d "id=$DEVICE_ID&auth_key=$SHELLY_KEY" | jq -r '.data.device_status."em:0" | del(.["user_calibrated_phase"]) | [.]' | \
	jq -r '(map(keys) | add | unique) as $cols | map(. as $row | $cols | map($row[.])) as $rows | $cols, $rows[] | @csv' > file.csv

while true
do
	curl -s -X POST $SHELLY_CLOUD/device/status -d "id=$DEVICE_ID&auth_key=$SHELLY_KEY" | jq -r '.data.device_status."em:0" | del(.["user_calibrated_phase"]) | [.]' | \
		jq -r '(map(keys) | add | unique) as $cols | map(. as $row | $cols | map($row[.])) as $rows | $rows[] | @csv' >> file.csv
	echo " $(date +'%d.%m.%Y') $(date +'%H:%M:%S.%3N') : row added"
	sleep 1
done