#date: 2021-10-25T16:58:27Z
#url: https://api.github.com/gists/09a6305399ed7e6a1fba68044d5faa24
#owner: https://api.github.com/users/kwanpham

#!/bin/bash -

# Created by https://github.com/kwanpham

#=-= START OF CUSTOM SERVICE CONFIGURATION =-#
# Where micro service war/jar file sits?
MS_HOME=/home/

MS_JAR=demo-0.0.1-SNAPSHOT.jar

MS_SERVICE=${MS_JAR%.*}

SHUTDOWN_WAIT=20 # before issuing kill -9 on process.

OPTIONS="";
#OPTIONS="-Dspring.config.location=/home/gbcoibft/deploy/resources/vch_citad_core.properties"
#=-= END OF CUSTOM CONFIGURATION =-=#

# Try to get PID of spring jar/war
MS_PID=$(ps fax | grep java | grep "${MS_JAR}" | awk '{print $1}')
export MS_PID

# Function: start
start() {

  if [ ! -f "$MS_JAR" ]; then
    echo "ERROR: ${MS_JAR} file not found"
    exit 1
  fi

  pid=${MS_PID}
  if [ -f "${pid}" ]; then
    {
      echo "Service : ${MS_JAR} is already running (pid: ${pid})"
    }
  else {
    # Start screener ms
    echo "Starting micro service ${MS_SERVICE}"
    nohup java -jar ${OPTIONS} ./${MS_JAR} >/dev/null 2>&1 &
    sleep 0.75
    echo "${MS_SERVICE} is running at pid: "$!

  }; fi

}

# Function: stop
stop() {
  pid=${MS_PID}
  if [ -n "${pid}" ]; then
    {

      #    run_as ${RUNASUSER} kill -TERM $pid
      kill -TERM $pid
      echo -ne "Stopping service module ${MS_SERVICE} "

      kwait=${SHUTDOWN_WAIT}

      count=0
      while kill -0 ${pid} 2>/dev/null && [ ${count} -le ${kwait} ]; do {
        printf "."
        sleep 1
        ((count++))
      }; done

      echo

      if [ ${count} -gt ${kwait} ]; then {
        printf "Process is still running after %d seconds, killing process" \
          ${SHUTDOWN_WAIT}
        kill ${pid}
        sleep 3

        # if it's still running use kill -9
        #
        if kill -0 ${pid} 2>/dev/null; then {
          echo "Process is still running, using kill -9"
          kill -9 ${pid}
          sleep 3
        }; fi
      }; fi

      if kill -0 ${pid} 2>/dev/null; then
        {
          echo "Error: process is still running, I give up"
        }
      else {
        # success, delete PID file, if you have used it with spring boot
        rm -f ${SPRING_BOOT_APP_PID}
      }; fi
    }
    echo "Stop ${pid} success"
  else {
    echo "Service : ${MS_SERVICE} is not running"
  }; fi

  #return 0;
}

# Function: status
status() {
  pid=$MS_PID
  if [ "${pid}" ]; then
    {
      port=$(ss -l -p -n | grep "pid=${pid}," | awk '{print $5}')

      echo "Service : ${MS_SERVICE} is running with pid: ${pid} and port ${port}"
    }
  else {
    echo "Service : ${MS_SERVICE} is not running"
  }; fi
}
# Main Code

case $1 in
start)
  start
  ;;
stop)
  stop
  ;;
restart)
  stop
  sleep 1
  start
  ;;
status)
  status
  ;;
*)
  echo
  echo "Usage: $0 { start | stop | restart | status }"
  echo
  exit 1
  ;;
esac

exit 0
