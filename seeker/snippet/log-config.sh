#date: 2023-01-17T16:52:41Z
#url: https://api.github.com/gists/f3ab1bfd431eccfab155f185da0702ac
#owner: https://api.github.com/users/AlexFernandes-MOVAI

#!/bin/bash

echo 'Configure systemd logging persistency on disk'

# Restore configuration to a predictable known state if a backup exists:
# if [ -f /etc/systemd/journald.conf.ORIGINAL ]; then
# 	mv /etc/systemd/journald.conf.ORIGINAL /etc/systemd/journald.conf
# fi


# Make a backup of the default config file- once taken all subsequent tests will fail so backup not overwritten
TIMESTAMP=$(date "+%Y%m%d-%H%M%S")

if [ ! -f "/etc/systemd/journald.conf.bak-$TIMESTAMP" ]; then
	cp -p /etc/systemd/journald.conf "/etc/systemd/journald.conf.bak-$TIMESTAMP"
fi

cat > /etc/systemd/journald.conf <<'EOF'
[Journal]
Compress=yes
# to store logs to disk in /var/log/journal/
Storage=Persistent
# Ensure logs do not eat all our storage space by expressly limiting their TOTAL disk usage:
SystemMaxUse=20G
# Stop writing log data even if below threshold specified in SystemMaxUse if total diskspace is running low using the *SystemKeepFree* directive:
SystemKeepFree=10%
# Limit the size log files can grow to before rotation
SystemMaxFileSize=10M
SystemMaxFiles=1000
SyncIntervalSec=1s
RateLimitIntervalSec=30s
# If more logs are received than what is specified in RateLimitBurst during the time interval defined by RateLimitIntervalSec,
# all further messages within the interval are dropped until the interval is over
RateLimitBurst=10000
# Purge log entries older than period specified in MaxRetentionSec directive
MaxRetentionSec=3months
# Rotate log no later than a week- if not already preempted by SystemMaxFileSize directive forcing a log rotation
MaxFileSec=1week
#Seal=yes
#SplitMode=uid
#RuntimeMaxUse=
#RuntimeKeepFree=
#RuntimeMaxFileSize=
#RuntimeMaxFiles=100
#ForwardToSyslog=yes
#ForwardToKMsg=no
#ForwardToConsole=no
#ForwardToWall=yes
#TTYPath=/dev/console
# Write only debug to disk
#MaxLevelStore=debug
# Write only debug to syslog
#MaxLevelSyslog=debug
# Max notification level to forward to the Kernel Ring Buffer (/var/log/messages)
#MaxLevelKMsg=notice
# Max notification level to forward to the console
#MaxLevelConsole=info
# Valid values for MaxLevelWall: emerg alert crit err warning notice info debug
#MaxLevelWall=emerg
#LineMax=48K
#ReadKMsg=yes
EOF

echo
echo "Changes made to /etc/systemd/journald.conf by script are $(tput setaf 1)RED$(tput sgr 0)"
echo "Original values are shown in $(tput setaf 2)GREEN$(tput sgr 0)"
echo
diff --color /etc/systemd/journald.conf "/etc/systemd/journald.conf.bak-$TIMESTAMP"
echo

systemctl daemon-reload

# Re-Read changes made to /etc/systemd/journald.conf
systemctl restart systemd-journald

systemctl status systemd-journald
