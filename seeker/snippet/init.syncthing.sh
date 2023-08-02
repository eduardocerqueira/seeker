#date: 2023-08-02T16:53:29Z
#url: https://api.github.com/gists/2e49194bc6ac0d8274574d23fab6af3b
#owner: https://api.github.com/users/ecxod

#!/bin/sh
### BEGIN INIT INFO
# Provides:          syncthing
# Required-Start:    $local_fs $remote_fs $network
# Required-Stop:     $local_fs $remote_fs $network
# Default-Start:     2 3 4 5
# Default-Stop:      0 1 6
# Short-Description: Syncthing
# Description:       Automatically sync files via secure, distributed technology
# Author:            Typomedia Foundation
# Version:           1.0
### END INIT INFO

PATH=/bin:/usr/bin:/sbin:/usr/sbin
DESC="Syncthing"
NAME=syncthing
DAEMON=/usr/bin/$NAME
PIDFILE=/var/run/$NAME.pid
TIMEOUT=60
USER=syncthing

[ -x $DAEMON ] || exit 0

. /lib/lsb/init-functions

check_daemon () {
    if [ $ENABLE_DAEMON != 1 ]; then
        log_progress_msg "(disabled)"
  	log_end_msg 255 || true
    else
        start_daemon
    fi
}

start_daemon () {
	start-stop-daemon --start --quiet \
		--make-pidfile --pidfile $PIDFILE \
		--background --chuid $USER \
		--exec $DAEMON
}

stop_daemon () {
	start-stop-daemon --stop --quiet \
		--pidfile $PIDFILE \
		--exec $DAEMON --retry $TIMEOUT \
		--oknodo
}

reload_daemon () {
	start-stop-daemon --stop --quiet \
		--exec $DAEMON \
		--oknodo --signal 1
}

case "$1" in
    start)
        log_daemon_msg "Starting $DESC"
        start_daemon
	log_end_msg 0
        ;;
    stop)
        log_daemon_msg "Stopping $DESC" "$NAME"
        stop_daemon
        log_end_msg 0
        ;;
    reload)
        log_daemon_msg "Reloading $DESC" "$NAME"
	reload_daemon
        log_end_msg 0
        ;;
    restart|force-reload)
        log_daemon_msg "Restarting $DESC"
	stop_daemon
	sleep 1
	start_daemon
	log_end_msg 0
        ;;
    status)
        status_of_proc "$DAEMON" "$DESC" && exit 0 || exit $?
        ;;
    *)
        log_action_msg "Usage: service $NAME {start|stop|reload|force-reload|restart|status}" || true
        exit 2
        ;;
esac

exit 0