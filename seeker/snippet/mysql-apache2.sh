#date: 2023-05-19T17:07:33Z
#url: https://api.github.com/gists/b4e144e8366f15a2007c8f411139edb5
#owner: https://api.github.com/users/remram44

#!/bin/sh

if [ "$(id -u)" != 0 ]; then echo "This script needs to run as root so that it can execute MySQL" >&2; exit 1; fi

# Start MySQL (in the background)
runuser -u mysql -- /usr/sbin/mysqld --pid-file=/run/mysqld/mysqld.pid &
sleep 5

# Need to set this to avoid apachectl talking to systemd
export APACHE_STARTED_BY_SYSTEMD=true

# Start Apache (in the background)
apachectl start

# Don't exit the whole script on Ctrl+C, do the graceful shutdown
trap ' ' INT

# Wait until we are done interacting with the site (user presses Ctrl+C)
sleep infinity
trap - INT

# Graceful shutdown
apachectl stop
/usr/bin/mysqladmin shutdown
