#date: 2022-05-18T17:07:22Z
#url: https://api.github.com/gists/ab02341c8d93105a4c34740105e00920
#owner: https://api.github.com/users/unref

#!/bin/sh
###
# description: stop and then start lxc container
# author:      unref (unref@inbox.ru)
# date:        18-05-2022
# license:     MIT
###


###
# global variables
###

CONTAINER_NAME=samba
STATUS_STOPPED=STOPPED
STATUS_RUNNING=RUNNING

###
# functions
###

container_status () {
  status=$(echo $(sudo lxc-info -H -s ${CONTAINER_NAME}))
  echo $status
}

start_container () {
  sudo lxc-start samba
}

stop_container () {
  sudo lxc-stop -W samba
}

###
# main
###

while [ "_$(container_status)" == "_${STATUS_RUNNING}" ]
do
  stop_container
  sleep .1
done

if [ "_$(container_status)" == "_${STATUS_STOPPED}" ]
then
  start_container
fi
