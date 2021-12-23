#date: 2021-12-23T17:06:33Z
#url: https://api.github.com/gists/d4a3fe45cdbbfc7a0f0eeafc1e85480f
#owner: https://api.github.com/users/jenkoian

#!/usr/bin/env bash

URL_TO_CHECK=${1}

if [ -z "$URL_TO_CHECK" ]; then
   echo 'Please supply a URL'
   exit 1;
fi

WP="wp"
GREEN='\033[0;32m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m' # No Color

echo "Finding random plugin to deactivate..."

PLUGINS=()
i=0
for plugin_name in $($WP plugin list --status=active --field=name); do
  PLUGINS[i]="$plugin_name"
  i=$((i+1))
done

size=${#PLUGINS[@]}
index=$(($RANDOM % $size))
random_plugin=${PLUGINS[$index]}

echo -e "Deactivating ${BOLD}$random_plugin${NC}"

$WP plugin deactivate $random_plugin --quiet

echo "Checking if site is still working..."

status=$(curl -L -s -k -o /dev/null -w "%{http_code}" "$URL_TO_CHECK");

if [ $status != 200 ]; then
    echo -e "${RED}Site breaks if you deactivate $random_plugin${NC}"
    exit=1
else
    echo -e "${GREEN}Site works if you deactivate $random_plugin${NC}"
    printf "${TICK}"
    exit=0
fi

echo -e "Reactivating ${BOLD}$random_plugin${NC}"

$WP plugin activate $random_plugin --quiet

exit $exit
