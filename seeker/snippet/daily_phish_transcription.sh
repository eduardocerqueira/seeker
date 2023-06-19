#date: 2023-06-19T16:43:25Z
#url: https://api.github.com/gists/48ad4a14b8d5619026bdfb0a12e201b7
#owner: https://api.github.com/users/joecannatti

#!/usr/bin/env bash
#
# setup on mac:
# brew install jq

PHISH_TOKEN= "**********"
DAY=`date +%m-%d`
API_BASE="http://phish.in/api/v1/"
DAILY_URL="$API_BASE/shows-on-day-of-year/$DAY"
METADATA_FILENAME="daily-phish-$DAY.json"
BEST_TRACK_FILENAME=best_track.json

mkdir -p -v this_day_in_phish/$DAY
cd this_day_in_phish/$DAY

echo "Downloading Daily Phish..."

curl $DAILY_URL \
  --output $METADATA_FILENAME \
  -H "Authorization: "**********"

if [[ $? == 0 ]]; then
  echo "CURL command to Metadada API Succeeded"
else
  echo "CURL command to Metadada API Failed"
  exit
fi

FETCH_SUCCESS=`cat $METADATA_FILENAME | jq '.success'`

if [[ $FETCH_SUCCESS != true ]]; then
  echo "API did not not accept our request. Exiting"
  exit
fi

# Choose the most liked show that happened on this day, and then the most liked song from that show

jq '.data | sort_by(.likes_count)[-1].tracks | sort_by(.likes_count)[-1]' $METADATA_FILENAME > $BEST_TRACK_FILENAME

AUDIO_URL=$(jq -r '.mp3' $BEST_TRACK_FILENAME)
AUDIO_FILENAME="$(jq -r '.title' $BEST_TRACK_FILENAME).mp3"

curl $AUDIO_URL --output $AUDIO_FILENAME

open $AUDIO_FILENAME -a 'Transcribe!'
open -a 'Dorico 4'rico 4'