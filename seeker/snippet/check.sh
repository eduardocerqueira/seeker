#date: 2023-01-11T17:05:36Z
#url: https://api.github.com/gists/ac378a9c632c235fa2033ba4c31f885f
#owner: https://api.github.com/users/temapskov

#!/bin/bash

#############################################################################
# 
# Use script: 
#    check-memleak.sh <check-execute-file>
# Return code: 0 - no memory leaks
#              1 - there is a memory leak
#
#############################################################################


exec_file=$1
report_file="$exec_file.report"

# echo "Start execute script"

valgrind --leak-check=full --log-file=$report_file $exec_file > /dev/null

ret=0
#echo "Result valgrind: $ret"

n=1
while read line; do

  # reading each line
  #echo "Line No. $n : $line"
  
  echo $line | grep -q "ERROR SUMMARY:"

  if [[ $? == 0 ]]
  then
    echo $line | grep -q -v "ERROR SUMMARY: 0 errors"
    if [[ $? == 0 ]]
    then
      ret=1
    fi
  fi

  n=$((n+1))
done < $report_file

rm -f $report_file

echo "Result mem leak - $ret"

exit $ret