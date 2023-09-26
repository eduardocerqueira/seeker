#date: 2023-09-26T16:52:57Z
#url: https://api.github.com/gists/7f8994ab651e58baad44ee2c3af13231
#owner: https://api.github.com/users/lynsei

#!/bin/bash
#
# error handling and error report example
#

set -e

# logfile
LOGFILE=`mktemp`

on_error()
{
  # restore
  exec 1>&3- 2>&4-

  # send email, syslog and so on
  echo "an error occured"
  echo "----------------"
  cat $LOGFILE
}

on_exit()
{
  # do something

  rm -f $LOGFILE
}

trap "on_error" ERR
trap "on_exit" EXIT


# save STDOUT, STDERR and redirect to logfile
exec 3>&1 4>&2
exec 1>$LOGFILE 2>&1

# do something
echo hoge
do_something_in_which_errors_may_occur
echo done

# restore STDOUT, STDERR and close temporary descriptor
exec 1>&3- 2>&4-
