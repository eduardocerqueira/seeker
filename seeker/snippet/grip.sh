#date: 2022-03-11T17:11:17Z
#url: https://api.github.com/gists/5720ccf92d83f1084fc4e77443547b03
#owner: https://api.github.com/users/leno3s

#!/bin/sh
#######################################
# grip.sh - Get RIven Prices shell by leno3s
#
# This is get and filter/sort weekly riven prices from DE Official API, for the game "Warframe".
# Output is json, so tv-cli ( github.com/uzimaru0000/tv ) is useful for viewing.
#
# This script requires curl, jq commands.
#
# example:
#  $ grip.sh // get all rivens as json
#  $ grip.sh | tv // get all prices as table view (sorted by weapon name in default)
#  $ grip.sh -t Melee -p | tv // get melee prices sorted by median
#  $ grip.sh -c Ogris | tv // get Ogris price
#
# Copyright (c) 2022 leno3s
# Published under the MIT Liscense
# https://opensource.org/licenses/mit-license.php
#######################################

usage() {
    echo "Command for Get weekly riven mod prices.
  Usage: grip [-t (Rifle|Pistol|Melee|Archgun|Zaw|Kitgun) -c 'Weapon Name' -r (true|false) -(a|m|p|s|h)]
    -t: Filter by Weapon types
    -c: Filter by Weapon name (Words must start with capital letter)
    -r: Filter by rolled status
    -a: Sort by average prices
    -m: Sort by median
    -p: Sort by population
    -s: Sort by standard deviation
    -h: View this help and exit"
}

response="$(curl -sL http://n9e5v4d8.ssl.hwcdn.net/repos/weeklyRivensPC.json)"
data="$response"

while getopts t:c:r:asmph opt
do
    case "$opt" in
        t) data=$(echo $data | jq "[.[] | select(contains({itemType: \"$OPTARG\"}))]")
            ;;
        c) if [ $OPTARG = "null" ]; then
            data=$(echo $data | jq "[.[] | select(.compatibility == null)]")
        else
            data=$(echo $data | jq "[.[] | select(contains({compatibility: \"$OPTARG\"}))]")
        fi
            ;;
        r) if "$OPTARG"; then
            data=$(echo $data | jq "[.[] | select(.rerolled)]")
        else
            data=$(echo $data | jq "[.[] | select(.rerolled|not)]")
        fi
            ;;
        a) data=$(echo $data | jq ". | sort_by(.avg)")
            ;;
        s) data=$(echo $data | jq ". | sort_by(.stddev)")
            ;;
        p) data=$(echo $data | jq ". | sort_by(.pop)")
            ;;
        m) data=$(echo $data | jq ". | sort_by(.median)")
            ;;
        h) usage && exit;;
    esac
done

echo $data
