#date: 2024-07-01T16:43:39Z
#url: https://api.github.com/gists/94fb761af1a2da4c9d0bd620d25c09d0
#owner: https://api.github.com/users/liquidz00

#!/bin/bash
#
# Author: liquidz00
# 	GitHub: https://github.com/liquidz00
#
# Written: Jul 01 24
#
# Following script prevents Teams status from going to idle by using
# 	osascript and System Events to mimic keystrokes every 5 mins
# 
# Use with caution, as script leverages caffeinate to continuously loop. It
# 	is recommended to copy and paste the script into your IDE for execution
#	instead of invoking from Terminal. 
#
# 

# Keep Teams alive
/usr/bin/caffeinate -d &
while true;
do
	/usr/bin/osascript -e 'tell application "Microsoft Teams" to activate'
	/usr/bin/osascript -e 'tell application "System Events" to keystroke "2" using {command down}'
	echo "Teams Status Refreshed"
	/bin/sleep 300
done