#date: 2023-05-26T17:00:36Z
#url: https://api.github.com/gists/9b633899b227bd6964daef469755fb6a
#owner: https://api.github.com/users/eliphaslevy

#!/bin/bash
MYNAME=$(basename $0)

# debug to stdout instead of syslog
if [ "${DEBUG:-0}" != "0" ] ; then
	logger() { echo "$(date) $@" ; }
fi

trap 'errno=$?; logger "$MYNAME PID $$ stop. errno:$errno" ; exit ${errno}' EXIT INT TERM

# exclusive lock
if [ "${MYLOCK:-}" != "locked" ]; then
	logger "$MYNAME start. PARENT=$$"
	lockdir=/var/run/$(basename $0)
	if [ ! -d "$lockdir" ]; then
		rm -f "$lockdir" 2>&1 || true
		mkdir "$lockdir" || exit 13
	fi
	export MYLOCK=locked
	export DEBUG=${DEBUG:-0}
	nice ionice flock --nonblock -E 100 -e "$lockdir" $SHELL $0 "$@"
	exit $?
fi

# personalization
email=youremail@example.com
dumpdir=/root/captures
dumpuser=root
dumpgroup=root

# interface to monitor
interface=$(ip route ls|sed -n '/default/{s/.* dev \([^ ]\+\).*/\1/;p;q;}')

packet_threshold=500000 # what is "too much bandwidth?"
buffer_size=30 # seconds of high bandwidth sustained, to trigger an email
log_packets=10000 # packets to log on tcpdump each time the alarm is triggered
log_timeout=15 # timeout for waiting the capture to finish - if the burst stops early

mail_wait=300 # one email each five minutes is ok
mail_counter=3 # but do not send more than 3 consecutive ones

# sent mail counter - is reset on each burst end
mails_sent=0

install -d -m 750 -o $dumpuser -g $dumpgroup $dumpdir

logger "$MYNAME start: CHILD=$$ INTERFACE=$interface THRESHOLD=$packet_threshold BUFFER=$buffer_size"

do_dump() {
	logger "$MYNAME high traffic: $pkt/s - starting a tcpdump"
	fname=$dumpdir/dump-$(date +"%Y%m%d_%H%M%S").cap
	timeout $log_timeout tcpdump -n -i $interface -c $log_packets -w "$fname"
	chown "$username:$groupname" "$fname"
	logger "$MYNAME packets dumped"
	if [ "$mails_sent" -gt "$mail_counter"  ] ; then # stop emails.
		logger "$MYNAME won't send email, $mails_sent consecutive ones"
		return
	fi
	logger "$MYNAME sending email alert"
	echo "$MYNAME captured file: $fname" | mail -s "PROBLEM: $pkt/s traffic on $HOSTNAME" "$email"
	mails_sent=$((mails_sent+1))
}

declare -a buffer
while true; do
	pkt_old=$(sed -n "/^$interface: /s/$interface: \([0-9]\+\) .*/\1/p" /proc/net/dev)
	sleep 1
	pkt_new=$(sed -n "/^$interface: /s/$interface: \([0-9]\+\) .*/\1/p" /proc/net/dev)

	pkt=$(( $pkt_new - $pkt_old ))
	[ "${DEBUG:-0}" == "1" ] && logger "$MYNAME $pkt/s"
	if [ $pkt -gt $packet_threshold ]; then
		logger "$MYNAME #${#buffer[*]} $pkt/s - above threshold."
		buffer+=($pkt)
		if [ "${#buffer[*]}" -ge "$buffer_size" ] ; then
			do_dump
			logger "$MYNAME sleeping $mail_wait seconds."
			sleep $mail_wait
			buffer=() # reset buffer
		fi
	else
		mails_sent=0 # reset mail counter
		buffer=() # reset buffer
	fi
done
