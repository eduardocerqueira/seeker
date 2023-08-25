#date: 2023-08-25T17:04:02Z
#url: https://api.github.com/gists/0308fde97e7f72381bfd6977b160d69e
#owner: https://api.github.com/users/jmagly

#!/bin/sh
# For vim4 ensure /boot/dtb/amlogic/kvim4.dtb.overlay.env has the following entry (reboot if change is needed)
# fdt_overlays=pwm_f

LEVEL=65000 # Turn on if temperature is over and off if its is under
LOOP_TIME=2 # Seconds between temperature check

# Initialize pwm
echo 1 | sudo tee /sys/class/pwm/pwmchip4/export
echo 1000000 | sudo tee /sys/class/pwm/pwmchip4/pwm1/period
echo 500000 | sudo tee /sys/class/pwm/pwmchip4/pwm1/duty_cycle

# Main loop
while true; do
  # Read Temperature
  cpu=$(cat /sys/class/thermal/thermal_zone0/temp)

  # Control fan speed
  echo $(date) ${cpu}

  if [ $cpu -lt $LEVEL ]; then
    echo 0 | sudo tee /sys/class/pwm/pwmchip4/pwm1/enable
  else
    echo 1 | sudo tee /sys/class/pwm/pwmchip4/pwm1/enable
  fi
  sleep $LOOP_TIME
done