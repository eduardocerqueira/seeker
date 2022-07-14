#date: 2022-07-14T17:10:14Z
#url: https://api.github.com/gists/966e3c1404d09dabebb469240c7a1f9d
#owner: https://api.github.com/users/swamirara

#!/bin/sh
# Make sure to populate the variables below with ones for your account

api_key=xxx
app_key=xxx
monitor_id=12345
monitor_scope=host:kratos

case "$1" in
    stop)
        # Mute monitor
        echo "Muting host monitor..."
        curl -X POST "https://app.datadoghq.com/api/v1/monitor/${monitor_id}/mute?api_key=${api_key}&application_key=${app_key}"
        ;;
    start)
        echo "Resolving and unmuting host monitor..."
        # Resolve monitor
        curl -X POST -H "Content-type: application/json" -d "{ \"resolve\": [{\"${monitor_id}\": \"${monitor_scope}\"}] }" "https://app.datadoghq.com/monitor/bulk_resolve?api_key=${api_key}&application_key=${app_key}"
        # Unmute monitor
        curl -X POST "https://app.datadoghq.com/api/v1/monitor/${monitor_id}/unmute?api_key=${api_key}&application_key=${app_key}"
        ;;
esac

exit 0