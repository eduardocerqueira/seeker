#date: 2022-04-21T17:03:01Z
#url: https://api.github.com/gists/078dc02346f48323cb6463fa4cb91a79
#owner: https://api.github.com/users/stren-12

#!/bin/sh
# Deployment script for codeigniter4 run this command as root (with sudo) to change the permissions
# To secure permissions and change the file/folder owner
# In linux most of the times you will have error say "Cache unable to write to ../writable/cache/"
# Note: in order to take advantage of this script you must run nginx, apache, etc... 
# Must run with www-data user and www-data group 
cp env .env
php spark key:generate
chmod 0700 -R writable/
chown www-data:www-data -R writable/
chmod 440 .env