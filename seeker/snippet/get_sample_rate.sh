#date: 2023-03-07T17:07:24Z
#url: https://api.github.com/gists/8560302758939cb5ff8fa78902a0e5bb
#owner: https://api.github.com/users/scoady

#!/bin/bash
##ln -sf /usr/local/bin/get_sample_rate <wherever_you_clone_this>/get_sample_rate.sh
## usage: get_sample_rate <customer org name>
## get_sample_rate GitHub
## https://app-meta.lightstep.com/lightstep-public/dashboard/j7y56JL6?selectedChart=Y61G6gHz&labels%5B0%5D%5Blabel_key%5D=org_name&labels%5B0%5D%5Blabel_value%5D=GitHub


function main() {
    export CUSTOMER_ORG="$1"
    export DASHBOARD_URL="https://app-meta.lightstep.com/lightstep-public/dashboard/j7y56JL6?selectedChart=Y61G6gHz&labels%5B0%5D%5Blabel_key%5D=org_name&labels%5B0%5D%5Blabel_value%5D=${CUSTOMER_ORG}"
    echo $DASHBOARD_URL
}


main "$@"