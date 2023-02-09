#date: 2023-02-09T17:06:52Z
#url: https://api.github.com/gists/425dea2f4ad5be7c66b37b51d978dffd
#owner: https://api.github.com/users/Jacse

#!/bin/bash
threshold=0.05
count=0

ten_min_check="up [0-9]{1} min"
while true
do
  load=$(uptime | sed -e 's/.*load average: //g' | awk '{ print $3 }')
  res=$(echo $load'<'$threshold | bc -l)

  if [[ uptime =~ $ten_min_check ]]
  then
    echo "First 10 minutes of boot"
  elif (( $res ))
  then
    echo "Idling.."
    ((count+=1))
  else
    echo "Working.."
    count=0
  fi
  echo "Idle minutes count = $count"

  if (( count>10 ))
  then
    echo Shutting down
    # wait a little bit more before actually pulling the plug
    sleep 120
    sudo poweroff
  fi

  sleep 60
done