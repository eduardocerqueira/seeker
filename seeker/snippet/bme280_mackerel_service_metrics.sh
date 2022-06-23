#date: 2022-06-23T17:01:35Z
#url: https://api.github.com/gists/26caac058a28c683676334519418021b
#owner: https://api.github.com/users/xruins

#!/usr/bin/env bash

SERVICE_NAME="Household"
API_KEY="2a587MJzUzs2vHoMxZeAEA7LgXN1Ys73b2Fy55LqwCjx"

# deploy bme280 script in specified path: https://github.com/SWITCHSCIENCE/BME280/blob/master/Python27/bme280_sample.py
SENSOR_METRICS=($(python2 /usr/local/bin/bme280_sample.py | grep -o -E '[0-9\.]+'))
TEMPERATURE=${SENSOR_METRICS[0]}
PRESSURE=${SENSOR_METRICS[1]}
HUMIDITY=${SENSOR_METRICS[2]}
NOW=$(date +%s)

REQUEST_BODY=$(cat <<-EOF
        [
                {"name": "temperature.verdandi", "time": $NOW, "value": $TEMPERATURE },
                {"name": "humidity.verdandi",    "time": $NOW, "value": $HUMIDITY },
                {"name": "pressure.verdandi",    "time": $NOW, "value": $PRESSURE }
        ]
EOF
)

curl https://api.mackerelio.com/api/v0/services/$SERVICE_NAME/tsdb \
    -H "X-Api-Key: $API_KEY" -H 'Content-Type: application/json' -X POST -d "$REQUEST_BODY"