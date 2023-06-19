#date: 2023-06-19T16:58:56Z
#url: https://api.github.com/gists/a65ec281b3b67cf71288a47cb82b6dc4
#owner: https://api.github.com/users/panwarnaresh

#!/bin/bash
# Script can be used to start / stop your entire fleet of VirtualBox VMs or to fetch list of running VMs
# Note all VMs you start with the script will be headless
# Place it as /usr/bin/mylab, chmod +x and use it like
# $ mylab start | $ mylab stop | $ mylab status
# ----------------

usage(){
echo "Usage: $0 {start|stop|status}"
}

status(){
  runningVMs=(`VBoxManage list runningvms | awk '{print $1}' | sed 's/\"//g'`)
  [[ ! -z ${runningVMs} ]] && echo -e "Running VMs are :\n${runningVMs[@]}" || echo -e "No VM is running!"
}

startlab(){
    installedVMs=`VBoxManage list vms | awk '{print $1}' | sed 's/\"//g'`
    for vm in $installedVMs
    do
    VBoxManage list runningvms | grep $vm
      [[ $? != 0 ]] && VBoxManage startvm $vm --type headless || echo -e "VM ${vm} is already running!"
    done
    echo -e "Started ... verifying in 10 seconds!\n"
    sleep 10
    status
}

stoplab(){
  runningVMs=(`VBoxManage list runningvms | awk '{print $1}' | sed 's/\"//g'`)
  if [ ! -z ${runningVMs} ]; then
    echo -e "VMs in running state: ${runningVMs[@]}"
    for vm in ${runningVMs[@]}
    do
      VBoxManage controlvm $vm poweroff
    done
  else
    echo "No VM is running!"
    exit 0
  fi
    echo -e "VMs were stopped ... verifying in 10 seconds!\n"
    sleep 10
    status
}

case "$1" in
  start)
    startlab
    ;;
  stop)
    stoplab
    ;;
  status)
    status
    ;;
  *)
    usage
esac
exit 0