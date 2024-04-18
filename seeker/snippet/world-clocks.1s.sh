#date: 2024-04-18T16:57:21Z
#url: https://api.github.com/gists/cc5d24eb559dd09d0ad594cb9ba93f76
#owner: https://api.github.com/users/jtzero

#!/usr/bin/env bash

# Display UTC in the menubar, and one or more additional zones in the drop down.
# The current format (HH:MM:SS) works best with a one second refresh, or alter
# the format and refresh rate to taste.
#
# <swiftbar.title>World Clocks</swiftbar.title>
# <swiftbar.version>v1.0</swiftbar.version>
# <swiftbar.author>Adam Snodgrass, jtzero</swiftbar.author>
# <swiftbar.author.github>jtzero</swiftbar.author.github>
# <swiftbar.desc>Display current UTC times in the menu bar</xbar.desc>
# <swiftbar.image></swiftbar.image>
# <swiftbar.var>string(VAR_ZONES="Australia/Sydney Europe/Amsterdam America/New_York America/Los_Angeles"): Space delimited set of timezones</swiftbar.var>

ZONES=("America/Chicago" "UTC")
for zone in "${ZONES[@]}"; do
  printf '%s' "$(TZ=$zone date +'%H:%M:%S %Z') "
done
printf '| font=Monaco'
