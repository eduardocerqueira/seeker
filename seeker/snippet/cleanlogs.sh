#date: 2023-09-29T17:04:02Z
#url: https://api.github.com/gists/e047b9932775510453b719ee8be67791
#owner: https://api.github.com/users/SA-JackMax

#!/bin/bash

# crontab
# 0 5 * * * /opt/alfresco/scripts/cleanlogs.sh

DAYS_TO_KEEP=7

find /opt/alfresco/logs/* -maxdepth 0 -name '*.log*' -mtime +${DAYS_TO_KEEP} -type f -exec rm {} \;
find /opt/alfresco/logs/bart/* -maxdepth 0 -name '*.log*' -mtime +${DAYS_TO_KEEP} -type f -exec rm {} \;
find /opt/alfresco/tomcat/logs/* -maxdepth 0 -name '*.log*' -mtime +${DAYS_TO_KEEP} -type f -exec rm {} \;