#date: 2024-03-22T16:57:47Z
#url: https://api.github.com/gists/9b0933ed15e04bd33364c59015d80bec
#owner: https://api.github.com/users/k8wu

#!/bin/bash
#
# zfsstats.sh - Shows ZFS ARC statistics refreshed every 2 seconds in a FreeBSD top(1)-like format
#
# Requires Linux with OpenZFS.

while true; do
  # capture ZFS ARC stats output, and populate/format needed variables
  zfs_stats=$(grep size /proc/spl/kstat/zfs/arcstats)
  declare -A raw_values
  declare -A formatted_values
  for var in size mfu_size mru_size anon_size hdr_size compressed_size uncompressed_size; do
    raw_values[${var}]=$(echo "${zfs_stats}" | egrep "^${var}" | awk '{print $3}')
    if [[ ${raw_values[${var}]} -ge 10485760 ]]; then
      formatted_values[${var}]="$(echo ${raw_values[${var}]} | awk '{print $1 / 1048576}' | cut -d. -f1)M"
    elif [[ ${raw_values[${var}]} -ge 10240 ]]; then
      formatted_values[${var}]="$(echo ${raw_values[${var}]} | awk '{print $1 / 1024}' | cut -d. -f1)K"
    else
      formatted_values[${var}]=${raw_values[${var}]}
    fi
  done

  # still need the "Other" value, which we get from subtracting the other variables from "size"
  raw_values['other']=$((${raw_values['size']} - ${raw_values['mfu_size']} - ${raw_values['mru_size']} - ${raw_values['anon_size]']} - ${raw_values['hdr_size']}))
  if [[ ${raw_values['other']} -ge 10485760 ]]; then
    formatted_values['other']="$(echo ${raw_values['other']} | awk '{print $1 / 1048576}' | cut -d. -f1)M"
  elif [[ ${raw_values[${var}]} -ge 10240 ]]; then
    formatted_values['other']="$(echo ${raw_values['other']} | awk '{print $1 / 1024}' | cut -d. -f1)K"
  else
    formatted_values['other']=${raw_values['other']}
  fi

  # we use awk to compute the ratio out to two decimal places
  formatted_values['ratio']=$(echo ${raw_values['uncompressed_size']} ${raw_values['compressed_size']} | awk '{printf "%.2f", $1 / $2}')

  # clear screen and print date/time
  clear
  date

  # output it all in the same format that FreeBSD top(1) uses for ZFS ARC stats
  echo "ARC: ${formatted_values['size']} Total, ${formatted_values['mfu_size']} MFU, ${formatted_values['mru_size']} MRU, ${formatted_values['anon_size']} Anon, ${formatted_values['hdr_size']} Header, ${formatted_values['other']} Other"
  echo "     ${formatted_values['compressed_size']} Compressed, ${formatted_values['uncompressed_size']} Uncompressed, ${formatted_values['ratio']}:1 Ratio"
  sleep 2
done
