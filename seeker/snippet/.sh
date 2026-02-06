#date: 2026-02-06T17:26:16Z
#url: https://api.github.com/gists/1bc233203966b9d1ef8a2768240f14e3
#owner: https://api.github.com/users/omgmog

prev=""
while true; do
  fb="$(fastboot devices 2>/dev/null | awk '/fastboot$/ {print $1; exit}')"
  ad="$(adb devices 2>/dev/null | awk 'NR>1 && NF {print $1":"$2; exit}')"
  pid="$(system_profiler SPUSBDataType | awk '
    /Product ID:/ {pid=$3}
    /Serial Number: 5A22000865/ {print pid; exit}
  ')"

  cur="pid=${pid:-na} fastboot=${fb:-none} adb=${ad:-none}"
  if [ "$cur" != "$prev" ]; then
    printf "%s  %s\n" "$(date '+%H:%M:%S')" "$cur"
    osascript -e 'beep 1'
    prev="$cur"
  fi
  sleep 1
done