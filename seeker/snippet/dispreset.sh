#date: 2022-05-18T16:58:49Z
#url: https://api.github.com/gists/17ca25bce8210018ccfa9e8a0b3398c8
#owner: https://api.github.com/users/altercation

#!/bin/sh

# dispreset:
# When called reset displays to last known arrangement using displayplacer
# if called within a timeout period (30 seconds), swap position.

# This is to work around the fact that in certain multimonitor configurations
# macOS fails to properly ID the displays using serial numbers pulled in via
# EDID and so loses track about which monitor is left/right positioned.

# E.g. my current setup involves two identical 4k monitors that are hooked up
# with a thunderbolt-to-dual-displaylink adapter.
# I call this utility on screen unlock using a mac utility called "EventScripts"

# Helpful info:
# view current displayplacer working command with:
# displayplacer list | grep displayplacer >> ~/bin/dispreset

# Requires:
# https://github.com/jakehilborn/displayplacer

TIMESTAMP=$(date +%s)
TIMESTAMP_LAST=0
WITHIN=30 # number of seconds
ID_B="id:BF4DD830-C664-479A-87F9-6F27FAE3B0D2"
ID_A="id:AB6C5F22-89F3-4AD5-90AF-F96FD093F524"
CONFIG=$HOME/.config/dispreset

set_defaults () {
	DISP_LEFT=$ID_A
	DISP_RIGHT=$ID_B
	DISP_LEFT_EXECUTE=$ID_A
	DISP_RIGHT_EXECUTE=$ID_B
}

if test -f "$CONFIG"; then
	. $CONFIG
else
	set_defaults
fi

if [ "$(($TIMESTAMP - $TIMESTAMP_LAST))" -lt "$WITHIN" ]; then
	#echo "< 5"
	#echo "CURRENT LEFT: $DISP_LEFT"
	#echo "CURRENT RIGHT: $DISP_RIGHT"
	DISP_LEFT_EXECUTE=$DISP_RIGHT
	DISP_RIGHT_EXECUTE=$DISP_LEFT
	#echo "NEW LEFT: $DISP_LEFT_EXECUTE"
	#echo "NEW RIGHT: $DISP_RIGHT_EXECUTE"
else
	#echo ">= 5"
	DISP_LEFT_EXECUTE=$DISP_LEFT
	DISP_RIGHT_EXECUTE=$DISP_RIGHT
fi

/opt/homebrew/bin/displayplacer \
"$DISP_LEFT_EXECUTE res:2560x1440 scaling:on origin:(0,0)" \
"$DISP_RIGHT_EXECUTE res:2560x1440 scaling:on origin:(1920,0)" \
|| set_defaults

echo "TIMESTAMP_LAST=$TIMESTAMP" > $CONFIG
echo "DISP_LEFT=\"$DISP_LEFT_EXECUTE\"" >> $CONFIG
echo "DISP_RIGHT=\"$DISP_RIGHT_EXECUTE\"" >> $CONFIG
