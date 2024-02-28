#date: 2024-02-28T17:06:20Z
#url: https://api.github.com/gists/46aa20417576cd3eebb4d3fba16bc4e0
#owner: https://api.github.com/users/mmoollllee

#!/bin/bash
# Read voltage from i2c if Wittypi is mounted
# run as `bash witty_voltage.sh`

readonly I2C_MC_ADDRESS=0x08
readonly I2C_VOLTAGE_IN_I=1
readonly I2C_VOLTAGE_IN_D=2

i2c_read()
{
  local retry=0
  if [ $# -gt 3 ]; then
    retry=$4
  fi
  local result=$(i2cget -y $1 $2 $3)
  if [[ "$result" =~ ^0x[0-9a-fA-F]{2}$ ]]; then
    echo $result;
  else
    retry=$(( $retry + 1 ))
    if [ $retry -ne 4 ]; then
      sleep 1
      #"I2C read $1 $2 $3 failed (result=$result), retrying $retry ..."
      i2c_read $1 $2 $3 $retry
    fi
  fi
}

calc()
{
  awk "BEGIN { print $*}";
}

i=$(i2c_read 0x01 $I2C_MC_ADDRESS $I2C_VOLTAGE_IN_I)
d=$(i2c_read 0x01 $I2C_MC_ADDRESS $I2C_VOLTAGE_IN_D)
calc $(($i))+$(($d))/100