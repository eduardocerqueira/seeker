#date: 2022-06-14T16:57:49Z
#url: https://api.github.com/gists/fd40e386ef073ee3c9fff7bbc2e5879f
#owner: https://api.github.com/users/shvchk

#! /usr/bin/env sh

host=""
wait=10
max_tries=3

# --------------------

log="logger -t wg-if-up"
host_up="false"
try=1

while :; do
  case $host_up in
    true)
      while ping -c 1 -W 1 $host &> /dev/null; do
        try=1
        sleep $wait
      done

      $log "No connection to $host detected (try $try)"

      if [ $try -lt $max_tries ]; then
        try=$(( try + 1 ))
        continue
      fi

      host_up="false"
      /etc/storage/wireguard/client.sh stop
      ;;

    false)
      until ping -c 1 -W 1 $host &> /dev/null; do
        sleep $wait
      done

      try=1
      host_up="true"
      /etc/storage/wireguard/client.sh start
      ;;
  esac

  $log "Host $host up: $host_up"
  sleep $wait
done
