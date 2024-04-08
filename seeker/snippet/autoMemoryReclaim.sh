#date: 2024-04-08T16:55:29Z
#url: https://api.github.com/gists/07457e4b76d9668a189ba24dfbcf24f7
#owner: https://api.github.com/users/fmg-cconley

#!/bin/bash

# Put under /etc/autoMemoryReclaim.sh

# set variables at the top
low_cpu_usage=50  # Note: We work with integer percentages (e.g., 50%)
idle_time=2       # Minutes
cached_memory_limit=1000  # MB
percent_memory_to_reclaim=5  # Percentage as an integer
wait_period=3

# function to check if user is idle
is_idle() {
  loadavg=$(cat /proc/loadavg)
  load1=$(echo "$loadavg" | awk '{print $1}')
  load5=$(echo "$loadavg" | awk '{print $2}')
  load15=$(echo "$loadavg" | awk '{print $3}')
  
  if (( $(echo "$load15 < $low_cpu_usage" | bc -l) )) && (( $(echo "$load1 < $low_cpu_usage" | bc -l) )) && (( $(echo "$load5 < $low_cpu_usage" | bc -l) )); then
    return 0
  else
    return 1
  fi
}

# function to get cached memory
get_cached_memory() {
  mem_info=$(free -m | grep "^Mem:" | awk '{print $6}')
  echo "$mem_info"
}

# initialize state
state="waiting_to_be_idle"
idle_counter=0

while true; do
  if [[ $state == "waiting_to_be_idle" ]]; then
     if is_idle; then
      ((idle_counter++))
      echo "User is idle for $idle_counter minutes"
      if ((idle_counter >= idle_time)); then
        state="reclaiming_memory"
        echo "User has been idle for $idle_time minutes, now moving to reclaiming memory state"
      fi
    else
      idle_counter=0
      echo "User is not idle"
    fi
  elif [[ $state == "reclaiming_memory" ]]; then
    if is_idle; then
      mem_cached=$(get_cached_memory)
      echo "Mem cached: $mem_cached ! and $cached_memory_limit"
      if [[ "$mem_cached" -lt "$cached_memory_limit" ]]; then
        state="waiting_to_be_idle"
        echo "Cached memory limit of $cached_memory_limit MB reached, now moving back to waiting to be idle state"
      else
        mem_total=$(awk '/MemTotal/ {print int($2/1024)}' /proc/meminfo)
        mem_to_reclaim=$((mem_total * percent_memory_to_reclaim / 100))
        echo "Reclaiming $mem_to_reclaim MB of memory"
        echo "${mem_to_reclaim}M" | sudo tee -a /sys/fs/cgroup/unified/memory.reclaim
        sleep $wait_period
      fi
    else
      state="waiting_to_be_idle"
      echo "User is not idle, now moving back to waiting to be idle state"
    fi
  fi

  # check every minute
  sleep $wait_period
done

