#date: 2022-05-05T16:52:16Z
#url: https://api.github.com/gists/ed6dfa39a433b2f517b19fd4c5a81c4a
#owner: https://api.github.com/users/rileyhales

#!/bin/bash

echo "Year chosen is $1"

mkdir "$1"
aws s3 cp s3://noaa-nwm-retrospective-2-1-pds/model_output/$1 ./$1 --no-sign-request --recursive --exclude "*" --include "*CHRTOUT_DOMAIN1.comp"

python nwm_to_daily.py "$1/"

mkdir "$1daily"
mv "$1*.nc" "$1daily"
zip -r "$1daily.zip" "$1daily"