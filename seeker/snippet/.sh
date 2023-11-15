#date: 2023-11-15T16:56:53Z
#url: https://api.github.com/gists/010fb978bae509dda43a1f31145a530f
#owner: https://api.github.com/users/dgalli1

#!/bin/bash

echo "Now Updating all Docker Containers"
export TZ=UTC # force all timestamps to be in UTC (+00:00 / Z)
printf -v start_date_epoch '%(%s)T'
printf -v start_date_iso8601 '%(%Y-%m-%dT%H:%M:%S+00:00)T' "$start_date_epoch"

# List of all folders that contain a docker compose
declare -a StringArray=("auth-stack" "bitwarden" "languagetool" "media-stack" "monitoring"  "" "mosquitto" "portainer" "dnsmasq" "socks5" "nginx-proxy-manager" "filebrowser")
# Iterate the string array using for loop
for val in ${StringArray[@]}; do
   cd /data
   echo Now Updating "$val"
   cd $val
   docker compose pull
   docker compose up -d
done
while IFS= read -r -d '' name; do
  names+=( "$name" )
done < <(
    docker container ls --format="{{.Names}}" | xargs -n1 docker container inspect | jq -j --arg start_date "$start_date_iso8601" '.[] | select(.State.StartedAt > $start_date) | (.Name, "\u0000")'
)
echo "now Updating the system"
apt update
apt upgrade

echo "Updated those containers:"
for containername in ${names[@]}; do
    echo "$containername"
done



if [ -f /var/run/reboot-required ] 
then
    echo "[*** Hello $USER, you must reboot your machine ***]"
fi
