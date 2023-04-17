#date: 2023-04-17T16:41:52Z
#url: https://api.github.com/gists/3448da5630c11e9b65a3d4d4248279fb
#owner: https://api.github.com/users/ostcrom

#!/bin/bash

##Script that uses inotify to check if changes have been made to any project files, if they are it reloads a docker-compose project.

#Check if WATCH_DIR and DOCKER_COMPOSE_FILE variables are set
if [ -z "${1}" ] || [ -z "${2}" ]; then
        echo "Usage: $0 WATCH_DIR DOCKER_COMPOSE_FILE"
        exit 1
fi

#make sure script is running with no-hup; the $nohup variable doesn't seem to get set as documentation suggests
if [ "${3}" != "nohup" ]; then
        echo "Restarting with nohup..."
        exec nohup "$0" "$@" "nohup" &
        exit
fi

#Set variables from the command line paramters
WATCH_DIR="${1}"
DOCKER_COMPOSE_FILE="${2}"

#We don't want to repeatedly restart the application, so we'll initialize a variable to keep track of the last run.
lastrun=$(( $(date +%s) - 180 ))

#docker-compose -f $DOCKER_COMPOSE_FILE up -d
inotifywait -r -m -e create,modify,delete --exclude ".git/*" $WATCH_DIR|
        while read path action file; do
                #Ignore changes to hidden files... for some reason I can't get the exclude in inotifywait to
                #towork as expected.
                if ! [[ $file =~ ^\..* ]];then
                        echo "$path$file $action"
                        now=$(date +%s)
                        if (( $now - $lastrun >= 180 )); then
                                echo "Initiating reload of $DOCKER_COMPOSE_FILE"
                                docker-compose -f $DOCKER_COMPOSE_FILE down
                                docker-compose -f $DOCKER_COMPOSE_FILE up -d
                                lastrun=$(date +%s)
                        else
                                echo "Last reload was less than three minutes ago, skipping..."
                        fi;
                fi
        done