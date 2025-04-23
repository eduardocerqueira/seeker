#date: 2025-04-23T17:08:14Z
#url: https://api.github.com/gists/66abdad211c204189a7f6f9b379f6560
#owner: https://api.github.com/users/uturuncoglu

#!/bin/bash

# get list of process
pid_lst=`ps -ef | grep $2 | awk '{print $2}'`

# loop over process and attach gdb
for i in $pid_lst
do
  # create prefix for config/log files in form of [machine]_[pid]
  prefix="${3}_${i}"

  # create temporary configuration file for gdb
  CONFFILE="$1/bt-${prefix}.conf"
  echo "set pagination off" >"$CONFFILE"
  echo "set logging file $1/bt-${prefix}.txt" >> "$CONFFILE" # write to specific file
  echo "set logging overwrite on" >> "$CONFFILE" # replace file
  echo "set logging redirect on" >> "$CONFFILE" # only goes to file
  echo "set logging on" >> "$CONFFILE" # log to file
  echo "attach $i" >> "$CONFFILE" # attach to process
  echo "bt" >> "$CONFFILE" # collect backtrace
  echo "detach" >> "$CONFFILE" # deattach to process
  echo "set logging off" >> "$CONFFILE" # disable logging
  echo "quit" >> "$CONFFILE" # quit

  # run gdb to collect bt
  echo "##### `hostname`:$i #####"
  gdb --batch -x "$CONFFILE" 2>/dev/null
  echo "##########"

  # remove temporary gdb configuration file
  rm -f $CONFFILE
done