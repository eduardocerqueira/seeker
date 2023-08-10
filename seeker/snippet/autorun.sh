#date: 2023-08-10T17:08:53Z
#url: https://api.github.com/gists/b6c9f11c160a286cc99c110aedebafbd
#owner: https://api.github.com/users/contributor

#!/bin/sh

# This script is to be used in combination with Synology Autorun:
# - https://github.com/reidemei/synology-autorun
# - https://github.com/Jip-Hop/synology-autorun
#
# You need to change the task_id to match your Hyper Backup task.
# Get it with command: more /usr/syno/etc/synobackup.conf
#
# I like to keep "Beep at start and end" disabled in Autorun, because I don't
# want the NAS to beep after completing (could be in the middle of the night)
# But beep at start is a nice way to confirm the script has started,
# so that's why this script starts with a beep.
#
# After the backup completes, the integrity check will start. 
# Unfortunately in DSM you can't choose to receive email notifications of the integrity check results.
# So there's a little workaround, at the end of this script, to send an (email) notification.
# The results of the integrity check are taken from the synobackup.log file.
#
# In DSM -> Control Panel -> Notification I enabled email notifications,
# I changed its Subject to %TITLE% and the content to:
# Dear user,
#
# Integrity check for %TASK_NAME% is done.
#
# %OUTPUT%
#
# This way I receive email notifications with the results of the Integrity Check.
#
# Credits:
# - https://github.com/Jip-Hop
# - https://bernd.distler.ws/archives/1835-Synology-automatische-Datensicherung-mit-DSM6.html
# - https://www.beatificabytes.be/send-custom-notifications-from-scripts-running-on-a-synology-new/

task_id=6 # Hyper Backup task id, get it with command: more /usr/syno/etc/synobackup.conf
task_name="USB3 3TB Seagate" # Only used for the notification

/bin/echo 2 > /dev/ttyS1 # Beep on start

startTime=$(date +"%Y/%m/%d %H:%M:%S") # Current date and time

device=$2 # e.g. sde1, passed to this script as second argument

# Backup
/usr/syno/bin/synobackup --backup $task_id --type image

while sleep 60 && /var/packages/HyperBackup/target/bin/dsmbackup --running-on-dev $device
do
    :
done

# Check integrity
/var/packages/HyperBackup/target/bin/detect_monitor -k $task_id -t -f -g
# Wait a bit before detect_monitor is up and running
sleep 60
# Wait until check is finished, poll every 60 seconds
/var/packages/HyperBackup/target/bin/detect_monitor -k $task_id -p 60

# Send results of integrity check via email (from last lines of log file)

IFS=''
output=""
title=
NL=$'\n'

while read line
do
    
    # Compute the seconds since epoch for the start date and time
    t1=$(date --date="$startTime" +%s)
    
    # Date and time in log line (second column)
    dt2=$(echo "$line" | cut -d$'\t' -f2)
    # Compute the seconds since epoch for log line date and time
    t2=$(date --date="$dt2" +%s)
    
    # Compute the difference in dates in seconds
    let "tDiff=$t2-$t1"
    
    # echo "Approx diff b/w $startTime & $dt2 = $tDiff"
    
    # Stop reading log lines from before the startTime
    if [[ "$tDiff" -lt 0 ]]; then
        break
    fi
    
    text=`echo "$line" | cut -d$'\t' -f4`
    # Get rid of [Local] prefix
    text=`echo "$text" | sed 's/\[Local\]//'`
    
    if [ -z ${title} ]; then
        title=$text
    fi
    
    output="$output${NL}$text"
    
done <<<$(tac /var/log/synolog/synobackup.log)

# Hijack the ShareSyncError event to send custom message.
# This event is free to reuse because I don't use the Shared Folder Sync (rsync) feature.
# More info on sending custom (email) notifications: https://www.beatificabytes.be/send-custom-notifications-from-scripts-running-on-a-synology-new/
/usr/syno/bin/synonotify "ShareSyncError" "{\"%OUTPUT%\": \"${output}\", \"%TITLE%\": \"${title}\", \"%TASK_NAME%\": \"${task_name}\"}"

# Sleep a bit more before unmounting the disk
sleep 60

# Unmount the disk
exit 100