#date: 2022-09-08T17:18:58Z
#url: https://api.github.com/gists/3a1c5f3300d33dd7889917c0ae18e990
#owner: https://api.github.com/users/nxjosephofficial

#!/bin/bash
# Add Youtube RSS
# for newsboat(text mode rss feed reader with podcast support)
template="https://www.youtube.com/feeds/videos.xml?channel_id="
echo "id?"
read id
if [ -z "$id" ]
then
      echo "id is empty"
      exit
else
	echo "$template""$id" >> $HOME/.newsboat/urls
fi