#date: 2022-07-08T16:56:14Z
#url: https://api.github.com/gists/2c2ed15bf2f7895d354a8e96be05b71d
#owner: https://api.github.com/users/lcrilly

#!/bin/bash
# unitc - a curl wrapper for configuring NGINX Unit
# [v1.0 09-Jul-2022] Liam Crilly <liam@nginx.com>

if [ $# -lt 1 ]; then
	echo "USAGE: ${0##*/} [HTTP method] [--quiet] URI"
        echo " - Configuration JSON is read from stdin"
        echo " - Default method is PUT with stdin, else GET"
        echo " - The control socket is automatically detected"
	exit 1
fi

# Defaults
#
QUIET=0
METHOD=PUT
SHOW_LOG=0

while [ $# -gt 1 ]; do
	case $1 in
		"-q" | "--quiet")
			QUIET=1
			shift
			;;

		"GET" | "PUT" | "POST" | "DELETE")
			METHOD=$1
			shift
			;;

		"HEAD" | "PATCH" | "PURGE" | "OPTIONS")
			echo "${0##*/} ERROR: Invalid HTTP method ($1)"
			exit 1
			;;

		*)
			echo "${0##*/} ERROR: Invalid option ($1)"
			exit 1
			;;
	esac
done

if [ "${1:0:1}" != "/" ]; then
    echo "${0##*/} ERROR: Invalid configuration URI"
    exit 1
fi

# Check if Unit is running, find the main process
#
PID=`ps x | grep unit:\ main | grep -v \ grep | awk '{print $1}'`
if [ "$PID" = "" ]; then
	echo "ERROR: unitd not running"
	exit 1
fi

# Read the significant unitd conifuration from cache file (or create it)
#
if [ -f /tmp/${0##*/}.$PID ]; then
	CONFIG=()
	while IFS= read -r line; do
		CONFIG+=("$line")
	done < /tmp/${0##*/}.$PID

	CURL_ADDR=${CONFIG[0]}
	ERROR_LOG=${CONFIG[1]}
else
	PARAMS=`ps -p $PID | grep unitd | cut -f2- -dv | tr '[]' ' ' | cut -f4- -d ' ' | sed -e 's/ --/\n--/g'`

	# Get control address
	#
	CTRL_ADDR=`echo $PARAMS | grep '\--control' | cut -f2 -d' '`
	if [ "$CTRL_ADDR" = "" ]; then
		CTRL_ADDR=`unitd --help | grep -A1 '\--control' | tail -1 |  cut -f2 -d\"`
	fi

	# Prepare for network or Unix socket addressing
	#
	if [ `echo $CTRL_ADDR | grep -c ^unix:` -eq 1 ]; then
		CURL_ADDR="--unix-socket `echo $CTRL_ADDR | cut -f2- -d:` _"
	else
		CURL_ADDR="http://$CTRL_ADDR"
	fi

	# Get error log filename
	#
	ERROR_LOG=`echo $PARAMS | grep '\--log' | cut -f2 -d' '`
	if [ "$ERROR_LOG" = "" ]; then
		ERROR_LOG=`unitd --help | grep -A1 '\--log' | tail -1 | cut -f2 -d\"`
	fi

	# Cache the discovery for this unit PID (and cleanup any old files)
	#
	rm /tmp/${0##*/}.* 2> /dev/null
	echo -en "${CURL_ADDR}\n${ERROR_LOG}\n" > /tmp/${0##*/}.$PID
fi

# Adjust HTTP method and curl params based on presence of stdin payload
#
LOG_LEN=`wc -l < $ERROR_LOG`
if [ -t 0 ]; then
	if [ "${1:0:8}" = "/control" ]; then
		SHOW_LOG=1
	fi
	curl -s $CURL_ADDR$1
else
	SHOW_LOG=1
	echo "$(cat)" | curl -s -X $METHOD --data-binary @- $CURL_ADDR$1
fi

if [[ $SHOW_LOG -eq 1 && $QUIET -eq 0 ]]; then
	echo -n "${0##*/}: Waiting for log..."
	sleep 1
	echo ""
	sed -n $((LOG_LEN+1)),\$p $ERROR_LOG
fi
