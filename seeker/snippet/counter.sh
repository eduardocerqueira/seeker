#date: 2024-06-27T17:06:49Z
#url: https://api.github.com/gists/934ae579510a1e3922fbd56db65f2141
#owner: https://api.github.com/users/meysam81

#!/bin/bash

total_seconds=0

while true
do
    total_seconds=$((total_seconds + 1))
    decades=$((total_seconds / 290304000))
    years=$((total_seconds / 29030400 % 10))
    months=$((total_seconds / 2419200 % 12))
    weeks=$((total_seconds / 604800 % 4))
    days=$((total_seconds / 86400 % 7))
    hours=$((total_seconds / 3600 % 24))
    minutes=$((total_seconds / 60 % 60))
    seconds=$((total_seconds % 60))

    time_string=""
    [ $decades -gt 0 ] && time_string="${time_string}${decades}d "
    [ $years -gt 0 ] && time_string="${time_string}${years}y "
    [ $months -gt 0 ] && time_string="${time_string}${months}m "
    [ $weeks -gt 0 ] && time_string="${time_string}${weeks}w "
    [ $days -gt 0 ] && time_string="${time_string}${days}d "
    [ $hours -gt 0 ] && time_string="${time_string}${hours}h "
    [ $minutes -gt 0 ] && time_string="${time_string}${minutes}m "
    [ $seconds -gt 0 ] && time_string="${time_string}${seconds}s"

    time_string=$(echo "$time_string" | sed 's/ *$//')

    echo "$time_string passed"


    sleep 1
done