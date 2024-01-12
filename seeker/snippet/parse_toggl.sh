#date: 2024-01-12T17:08:07Z
#url: https://api.github.com/gists/5ccfdff391f1bfc894f6e21beb78996d
#owner: https://api.github.com/users/alifeee

#!/bin/bash

# turns Toggl JSON API data into CSV format
#  currently ignores any array fields
#  you can override the TOGGL_KEYS env variable to set which keys you want in the CSV
# usage
#   get toggl data
#   https://developers.track.toggl.com/docs/api/time_entries/index.html
#   curl  https: "**********": application/json"   -u <email>:<password> > toggl.json
#   script:
#   as argument:
#     ./parse_toggl.sh toggl.json
#   as pipe
#     cat toggl.json | /parse_toggl.sh
#   or just directly pipe curl output to the script (use "-s" with curl to suppress loading output to stderr)

KEYS="${TOGGL_KEYS:-.id, .workspace_id, .project_id, .task_id, .billable, .start, .stop, .duration, .description, .duronly, .at, .server_deleted_at, .user_id, .uid, .wid, .pid}"

# print CSV headers
echo $KEYS | awk -F', ' '{for (i=1; i<NF; i++) {printf "%s,", substr($i, 2)}} END {printf "\n"}'

# print CSV content (parse JSON with jq)
cat $1 | jq -r '.[] | ['"${KEYS}"'] | @csv'th jq)
cat $1 | jq -r '.[] | ['"${KEYS}"'] | @csv'