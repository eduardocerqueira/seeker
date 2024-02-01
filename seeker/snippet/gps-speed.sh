#date: 2024-02-01T16:56:19Z
#url: https://api.github.com/gists/19ab477ba43b685103c107d1cbb1dc34
#owner: https://api.github.com/users/tve

#! /bin/bash
active=$(systemctl is-active gpsd.service)
if [[ "$active" == active ]]; then
  echo "Stopping gpsd"
  sudo systemctl stop gpsd.service
  sudo systemctl stop gpsd.socket
  sleep 1
fi

function cleanup {
  stty -F /dev/ttyGPS ispeed 9600 ospeed 9600
  if [[ "$active" == active ]]; then
    echo "Restarting gpsd"
    sudo systemctl start gpsd.service
    sudo systemctl start gpsd.socket
  fi
}

for s in 9600 19200 38400 4800 57800; do
  echo "Trying $s"
  stty -F /dev/ttyGPS ispeed $s ospeed $s
  #timeout 2 tail -f /dev/ttyGPS || true
  if timeout 3 tail -f /dev/ttyGPS | grep -q '^\$GPRMC,'; then
    echo "Device runs at $s baud"
    if [[ $s != 9600 ]]; then
      echo "Switching to 9600"
      #timeout 3 tail -f /dev/ttyGPS &
      # gpsctl --echo --type MTK-3301 -x '$PMTK251,9600'
      printf '$PMTK251,9600*17\r\n' >/dev/ttyGPS
      sleep 1
    fi
    cleanup
    exit 0
  fi
done
echo "Unknown speed"
stty -F /dev/ttyGPS ispeed 9600 ospeed 9600
timeout 3 tail -f /dev/ttyGPS | od -c
cleanup
exit 1
