#date: 2022-03-15T17:08:59Z
#url: https://api.github.com/gists/0a98a84d5a0b4617e943589e23de019e
#owner: https://api.github.com/users/frullah

# shutdown until process finishes
# 
## arguments
# first argument is a PID

pid=$1

while true
do
  if ! ps -p $pid > /dev/null
  then
    sudo shutdown -h now
  fi

  sleep 5
done