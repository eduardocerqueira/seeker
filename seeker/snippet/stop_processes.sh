#date: 2024-07-01T17:02:02Z
#url: https://api.github.com/gists/caa0df680f38f02bbdea8db40e9e0eab
#owner: https://api.github.com/users/alamindevms

#file: stop_processes.sh

# List of process names to be stopped
PROCESS_NAMES=("chrome")

for PROCESS_NAME in "${PROCESS_NAMES[@]}"
do
  PIDS=$(pgrep $PROCESS_NAME)
  echo "Name - $PROCESS_NAME | PIDS - $PIDS"

  for PID in $PIDS
  do
    kill $PID

    if kill -0 $PID &> /dev/null; then
      kill -9 $PID
      echo "Killed $PID"
    fi
  done
done