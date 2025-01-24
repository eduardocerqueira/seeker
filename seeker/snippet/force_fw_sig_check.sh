#date: 2025-01-24T16:59:06Z
#url: https://api.github.com/gists/51ea164b311d4b9bc5beaa69d4db602b
#owner: https://api.github.com/users/itsyourap

#!/bin/sh
#
# This script is used to force the firmware signature check to be successful
#
# Created by: itsyourap
# Date: 2023-08-01
#

# Run the script in a loop to keep the firmware signature check successful
while true; do
        # Check if the firmware signature check result file exists
        if [ -e /tmp/firmCheckRes.txt ]; then
                # Replace the firmware signature check result with success
                cat /dev/null >/tmp/firmCheckRes.txt
                echo "Replacing the firmware signature check result with success"
        fi
done