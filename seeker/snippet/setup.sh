#date: 2022-10-27T17:13:14Z
#url: https://api.github.com/gists/54111cad1cdcb3c4a0ce27fb890f3b74
#owner: https://api.github.com/users/josephfusco

#!/usr/bin/env bash

# Exit if WP-CLI is not available.
if ! command -v wp &> /dev/null
then
    echo "This script requires WP-CLI"
    exit
fi

# Install WordPress plugins.
wp plugin install atlas-content-modeler --activate
wp plugin install wordpress-importer --activate

# Modify plugin data.
wp acm reset
wp acm blueprint import demo