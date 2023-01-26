#date: 2023-01-26T17:06:59Z
#url: https://api.github.com/gists/a981647422333c705c4a1932e7d36b4f
#owner: https://api.github.com/users/EmberHeartshine

#!/bin/bash

# SYSTEM REQUIREMENTS:
#  - python
#  - jq
#  - yq (pip install yq)

# Some notes: Because this script uses Nitter as a scraper, it cannot handle video posts.
# Multi-image/album posts are a WIP and are not working. (only the first image is scraped)
# This script is tested working with crontab, assuming a standard python/pip install.

# User-friendly name (can be anything, but should not share with others of the same script)
NAME=

# Your Mastodon instance
INSTANCE=

# Mastodon access token; get it from Preferences > Development > $YOUR_APPLICATION
# If you haven't made an application, do so now and get the access token.
# The application MUST have the following permissions: write:media write:statuses
AUTH=

# The Twitter account to monitor
BIRDACCT=

# The Nitter instance to use (default should be fine unless you have any reason to use another instance)
NITTER=nitter.fly.dev

cd $(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
PATH=$PATH":/home/$(whoami)/.local/bin"

if test -f $NAME'_pubdate.txt'; then
	RSS=$(curl -s -L http://$NITTER/$BIRDACCT/rss)
	POSTDATE=$(echo $RSS |xq -r '.rss.channel.item[0].pubDate')
	if [[ "$(cat $NAME'_pubdate.txt')" == "$POSTDATE" ]]; then
		echo "No new $NAME post to fetch."
	else
		if [[ ! -z $(echo $RSS |xq '.rss.channel.item[0].description' |sed 's/\\n/\n/g' |grep img) ]]; then
			echo "Found new image post for $NAME. Fetching and uploading to Mastodon."
			FIELDNUM=${NITTER//[^\.]}
			FIELDNUM=$(( 2 + ${#FIELDNUM} ))
			IMAGE=$(echo $RSS |xq '.rss.channel.item[0].description' |sed 's/ /\n/g' |grep src |cut -d '\' -f 2 |cut -c2- |head -1)
			EXT=$(echo $IMAGE |cut -d '.' -f $FIELDNUM)
			curl -s -L $IMAGE -o $NAME.$EXT
			IMGID=$(curl -s -H "Authorization: Bearer "$AUTH -X POST -H 'Content-Type: multipart/form-data' https://$INSTANCE/api/v2/media --form file="@"$NAME.$EXT |jq -r '.id')
			TEXT=$(echo $RSS |xq -r '.rss.channel.item[0].title')
			TEXT=$(echo $TEXT |sed s/^@//g)
			curl -s https://$INSTANCE/api/v1/statuses -H "Authorization: Bearer "$AUTH -F "status=$TEXT" -F "media_ids[]="$IMGID |jq -r '.url' >> $NAME'_entries.log'
			rm $NAME.$EXT
		else
			echo "Found new text post for $NAME. Mirroring to Mastodon."
			TEXT=$(echo $RSS |xq -r '.rss.channel.item[0].title')
			TEXT=$(echo $TEXT |sed s/^@//g)
			curl -s https://$INSTANCE/api/v1/statuses -H "Authorization: Bearer "$AUTH -F "status=$TEXT" |jq -r '.url' >> $NAME'_entries.log'
		fi
		echo $POSTDATE > $NAME'_pubdate.txt'
	fi
else
	echo 0 > $NAME'_pubdate.txt'
	./${0##*/}
fi