#date: 2021-10-29T17:05:54Z
#url: https://api.github.com/gists/dcc1e836197b4f4311ecaabc16e1a818
#owner: https://api.github.com/users/emmcdonald

#!/bin/sh
# Source from here:
# https://gist.github.com/SeonghoonKim/5385982

# Michael.McDonald@hycom.org
# Date Oct 29, 2021
# Added "drain" option.

# Set up a default search path
PATH="/usr/bin:/bin"

CURL=`which curl`
if [ -z "$CURL" ]; then
  echo "curl not found"
  exit 1
fi

proto='http'
server='localhost'
port=80
manager='balancer-manager'
auth='-u USER:PASS'
debug='-s' # default keep on silent mode 

while getopts "s:p:m:a:d" opt; do
  case "$opt" in
    s)
      server=$OPTARG
      ;;
    p)
      port=$OPTARG
      ;;
    m)
      manager=$OPTARG
      ;;  
    a)
      auth=$OPTARG
      ;;  
    d)
      debug=$OPTARG
      ;;  
  esac
done

shift $(($OPTIND - 1))
action=$1

list_balancers() {
  $CURL $auth $debug "http://${server}:${port}/${manager}" | grep "balancer://" | sed "s/.*balancer:\/\/\(.*\)<\/a>.*/\1/"
}

list_workers() {
  balancer=$1
  if [ -z "$balancer" ]; then
    echo "Usage: $0 [-s host] [-p port] [-m balancer-manager]  list-workers  balancer_name"
    echo "  balancer_name :    balancer name"
    exit 1
  fi  
  $CURL $auth $debug "http://${server}:${port}/${manager}" | grep "/${manager}?b=${balancer}&amp;w" | sed "s/.*href='\(.[^']*\).*/\1/" | sed "s/.*w=\(.*\)&.*/\1/"
}

enable() {
  balancer=$1
  worker=$2
  if [ -z "$balancer" ] || [ -z "$worker" ]; then
    echo "Usage: $0 [-s host] [-p port] [-m balancer-manager]  enable  balancer_name  worker_route"
    echo "  balancer_name :    balancer/cluster name"
    echo "  worker_route  :    worker route e.g.) ajp://192.1.2.3:8009"
    exit 1
  fi
  
  nonce=`$CURL $auth $debug "http://${server}:${port}/${manager}" | grep nonce | grep "${balancer}" | sed "s/.*nonce=\(.*\)['\"].*/\1/" | tail -n 1`
  if [ -z "$nonce" ]; then
    echo "balancer_name ($balancer) not found"
    exit 1
  fi

  ## 2021-10-29 setting the Referer is needed with Apache 2.4.x
  echo "Enabling $2 of $1..."
  #enable=on
  $CURL $auth $debug -o /dev/null -XPOST "http://${server}:${port}/${manager}?" -H "Referer: http://${server}:${port}/${manager}?b=${balancer}&w=${worker}&nonce=${nonce}" -d b="${balancer}" -d w="${worker}" -d nonce="${nonce}" -d w_status_D=0
  #drain=off
  $CURL $auth $debug -o /dev/null -XPOST "http://${server}:${port}/${manager}?" -H "Referer: http://${server}:${port}/${manager}?b=${balancer}&w=${worker}&nonce=${nonce}" -d b="${balancer}" -d w="${worker}" -d nonce="${nonce}" -d w_status_N=0
  sleep 2
  status
}

disable() {
  balancer=$1
  worker=$2
  if [ -z "$balancer" ] || [ -z "$worker" ]; then
    echo "Usage: $0 [-s host] [-p port] [-m balancer-manager]  disable  balancer_name  worker_route"
    echo "  balancer_name :    balancer/cluster name"
    echo "  worker_route  :    worker route e.g.) ajp://192.1.2.3:8009"
    exit 1
  fi
  
  echo "Disabling $2 of $1..."
  nonce=`$CURL $auth $debug $curl_flags "http://${server}:${port}/${manager}" | grep nonce | grep "${balancer}" | sed "s/.*nonce=\(.*\)['\"].*/\1/" | tail -n 1`
  if [ -z "$nonce" ]; then
    echo "balancer_name ($balancer) not found"
    exit 1
  fi

  ## 2021-10-29 setting the Referer is needed with Apache 2.4.x
  $CURL $auth $debug -o /dev/null -XPOST "http://${server}:${port}/${manager}?" -H "Referer: http://${server}:${port}/${manager}?b=${balancer}&w=${worker}&nonce=${nonce}" -d b="${balancer}" -d w="${worker}" -d nonce="${nonce}" -d w_status_D=1
  sleep 2
  status
}

drain() {
  balancer=$1
  worker=$2
  if [ -z "$balancer" ] || [ -z "$worker" ]; then
    echo "Usage: $0 [-s host] [-p port] [-m balancer-manager]  drain  balancer_name  worker_route"
    echo "  balancer_name :    balancer/cluster name"
    echo "  worker_route  :    worker route e.g.) ajp://192.1.2.3:8009"
    exit 1
  fi
  
  echo "Draining $2 of $1..."
  nonce=`$CURL $auth $debug $curl_flags "http://${server}:${port}/${manager}" | grep nonce | grep "${balancer}" | sed "s/.*nonce=\(.*\)['\"].*/\1/" | tail -n 1`
  if [ -z "$nonce" ]; then
    echo "balancer_name ($balancer) not found"
    exit 1
  fi

  ## 2021-10-29 setting the Referer is needed with Apache 2.4.x
  $CURL $auth $debug -o /dev/null -XPOST "http://${server}:${port}/${manager}?" -H "Referer: http://${server}:${port}/${manager}?b=${balancer}&w=${worker}&nonce=${nonce}" -d b="${balancer}" -d w="${worker}" -d nonce="${nonce}" -d w_status_N=1
  sleep 2
  status | grep $worker
}

status() {
  $CURL $auth $debug "http://${server}:${port}/${manager}" | grep "a href" | sed "s/<[^>]*>/ /g"
}

case "$1" in
  list-balancer)
    list_balancers "${@:2}"
	;;
  list-worker)
    list_workers "${@:2}"
	;;
  enable)
    enable "${@:2}"
	;;
  disable)
    disable "${@:2}"
	;;
  drain)
    drain "${@:2}"
	;;
  status)
    status "${@:2}"
	;;
  *)
    echo "Usage: $0 {list-balancer|list-worker|enable|disable|status}"
	echo ""
	echo "Options: "
	echo "    -s server"
	echo "    -p port"
	echo "    -m balancer-manager-context-path"
	echo ""
	echo "Commands: "
	echo "    list-balancer"
	echo "    list-worker  balancer-name"
	echo "    enable   balancer_name  worker_route"
	echo "    disable  balancer_name  worker_route"
	echo "    drain    balancer_name  worker_route"
    exit 1
esac

exit $?
