#date: 2022-10-25T17:36:43Z
#url: https://api.github.com/gists/9bfbb94efb5228c8c793e3bebc950d48
#owner: https://api.github.com/users/maisilex

#!/bin/bash

# Make sites into bedrock or normal WordPress
# Version 1.0
# Copyright (c) NeONBRAND

set -a
source .env
set +a

if [[ "$*" == *"--more"* ]]
then
    WP=true
    echo "👎 Convert to Vanilla WP"
else
    WP=false
fi

if [[ "$*" == *"--less"* ]]
then
    BEDROCK=true
    echo "👍 Convert to Bedrock"
else
    BEDROCK=false
fi

if [[ "$WP" = true ]]; then

    # Set our variables that we need to run our scripts
    echo "*** This script must run from the a folder adjacent to your site folder (ie. /code/website/ && /code/script/) ***"
    read -rp 'Enter the folder name of the website to be converted: ' SITENAME

    cd ../ &&
    mkdir "${SITENAME}-wp" &&
    echo "*** Folder ${SITENAME}-wp created ***" &&
    cd "${SITENAME}-wp"
    wp core download
    wp config create --dbname=vwp_$SITENAME --dbuser=root --dbpass=$MYSQL_ROOT_PWD
    echo "*** Syncing content folder over, this may take a minute ***"
    rsync -rz ../$SITENAME/web/app/ wp-content
    rm -rf wp-content/mu-plugins/*
    cd ../$SITENAME &&
    wp db export ../${SITENAME}-wp/wp-content/dbbackup.sql

    # Create the database if it isn't already made
    if [[ -z "$MYSQL_ROOT_PWD" ]]
    then
        if ! $(command -v mysql) -u root -e "use vwp_$SITENAME"
        then
            $(command -v mysql) -uroot -e "CREATE DATABASE vwp_${SITENAME} /*\!40100 DEFAULT CHARACTER SET utf8mb4 */;"
            $(command -v mysql) -uroot -e "FLUSH PRIVILEGES;"
            echo -e "Created vwp_$SITENAME database."
        else
            echo "The \"vwp_$SITENAME\" database already exists. Continue? "
            options=("Proceed" "Drop database and proceed" "Quit")
            select opt in "${options[@]}"
            do
                case $opt in
                    "Proceed")
                        echo "Continuing without modifying \"vwp_$SITENAME\"..."
                        break
                        ;;
                    "Drop database and proceed")
                        # TODO Maybe a macOS popup to confirm?
                        echo Y | $(command -v mysqladmin) -uroot drop "vwp_$SITENAME"
                        $(command -v mysql) -uroot -e "CREATE DATABASE vwp_${SITENAME} /*\!40100 DEFAULT CHARACTER SET utf8mb4 */;"
                        $(command -v mysql) -uroot -e "FLUSH PRIVILEGES;"
                        echo -e "Created vwp_$SITENAME database."
                        break
                        ;;
                    "Quit")
                        echo "Terminated by user. Deleting project files."
                        $(command -v rm) -rf "$SITENAME"
                        exit 0
                        ;;
                    *) echo "Try again -- $REPLY is not an option";;
                esac
            done
        fi
    else
        if ! $(command -v mysql) -u root -p"$MYSQL_ROOT_PWD" -e "use $SITENAME"
        then
            $(command -v mysql) -uroot -p"$MYSQL_ROOT_PWD" -e "CREATE DATABASE ${SITENAME} /*\!40100 DEFAULT CHARACTER SET utf8mb4 */;"
            $(command -v mysql) -uroot -p"$MYSQL_ROOT_PWD" -e "FLUSH PRIVILEGES;"
            echo -e "Created $SITENAME database."
        else
            echo -e "Could not create $SITENAME database. Please delete it and try again."
            exit 1
        fi
    fi
    # echo $PWD
    cd ../${SITENAME}-wp &&
    chmod -R 755 wp-content/uploads/ &&
    wp db import wp-content/dbbackup.sql &&
    wp search-replace "$SITENAME" "$SITENAME-wp" &&
    wp search-replace "/app/" "/wp-content/" &&
    wp option update home "http://${SITENAME}-wp.test" &&
    wp option update siteurl "http://${SITENAME}-wp.test"
fi
