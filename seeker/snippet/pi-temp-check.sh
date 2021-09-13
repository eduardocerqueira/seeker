#date: 2021-09-13T17:16:13Z
#url: https://api.github.com/gists/07d1d8f3f604c5b74798ec9c2c19bd32
#owner: https://api.github.com/users/ellartdev

#!/bin/bash
# Script for checking CPU and GPU temperature
# ===========================================================
cpu_temp=$(</sys/class/thermal/thermal_zone0/temp)

echo "$(date) @ $(hostname)"
echo "============================================="
echo "GPU => $(/opt/vc/bin/vcgencmd measure_temp)"
echo "CPU => $((cpu_temp/1000))'C"
