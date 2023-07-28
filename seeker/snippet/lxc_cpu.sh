#date: 2023-07-28T16:37:44Z
#url: https://api.github.com/gists/12253c66bfcbd84d1e404d5e59e57e9c
#owner: https://api.github.com/users/jens-maus

#!/bin/bash 

cpu_last_sum=0
while :; do
  # get the first line with aggregate of all CPUs 
  cpu_now=($(head -n1 /proc/stat)) 

  # get all columns but skip the first (which is the "cpu" string) 
  cpu_sum="${cpu_now[@]:1}" 

  # replace the column seperator (space) with + 
  cpu_sum=$((${cpu_sum// /+})) 

  # check if we have cpu_last_sum already
  if [[ ${cpu_last_sum} -gt 0 ]]; then

  	# get the delta between two reads 
  	cpu_delta=$((cpu_sum - cpu_last_sum)) 
 
  	# get the idle time Delta 
  	cpu_user=$((cpu_now[1]- cpu_last[1])) 
  	cpu_nice=$((cpu_now[2]- cpu_last[2])) 
  	cpu_system=$((cpu_now[3]- cpu_last[3])) 
  	cpu_idle=$((cpu_now[4]- cpu_last[4])) 
  	cpu_iowait=$((cpu_now[5]- cpu_last[5])) 
  	cpu_irq=$((cpu_now[6]- cpu_last[6])) 
  	cpu_softirq=$((cpu_now[7]- cpu_last[7])) 
  	cpu_steal=$((cpu_now[8]- cpu_last[8])) 
  	cpu_guest=$((cpu_now[9]- cpu_last[9])) 
  	cpu_guest_nice=$((cpu_now[10]- cpu_last[10])) 

  	# calc time spent working 
  	cpu_used=$((cpu_delta - cpu_idle)) 
  	cpu_user=$((cpu_delta - cpu_user)) 
  	cpu_nice=$((cpu_delta - cpu_nice)) 
  	cpu_system=$((cpu_delta - cpu_system)) 
  	cpu_idle=$((cpu_delta - cpu_idle)) 
  	cpu_iowait=$((cpu_delta - cpu_iowait)) 
  	cpu_irq=$((cpu_delta - cpu_irq)) 
  	cpu_softirq=$((cpu_delta - cpu_softirq)) 
  	cpu_steal=$((cpu_delta - cpu_steal)) 
  	cpu_guest=$((cpu_delta - cpu_guest)) 
  	cpu_guest_nice=$((cpu_delta - cpu_guest_nice)) 

  	# calc percentage using integer arithmetic
	cpu_usage=$(printf %.2f "$((10000000 * (100 * cpu_used) / cpu_delta))e-7")
	cpu_user=$(printf %.2f "$((1000000000 - (10000000 * (100 * cpu_user) / cpu_delta)))e-7")
	cpu_nice=$(printf %.2f "$((1000000000 - (10000000 * (100 * cpu_nice) / cpu_delta)))e-7")
	cpu_system=$(printf %.2f "$((1000000000 - (10000000 * (100 * cpu_system) / cpu_delta)))e-7")
	cpu_idle=$(printf %.2f "$((1000000000 - (10000000 * (100 * cpu_idle) / cpu_delta)))e-7")
	cpu_iowait=$(printf %.2f "$((1000000000 - (10000000 * (100 * cpu_iowait) / cpu_delta)))e-7")
	cpu_irq=$(printf %.2f "$((1000000000 - (10000000 * (100 * cpu_irq) / cpu_delta)))e-7")
	cpu_softirq=$(printf %.2f "$((1000000000 - (10000000 * (100 * cpu_softirq) / cpu_delta)))e-7")
	cpu_steal=$(printf %.2f "$((1000000000 - (10000000 * (100 * cpu_steal) / cpu_delta)))e-7")
	cpu_guest=$(printf %.2f "$((1000000000 - (10000000 * (100 * cpu_guest) / cpu_delta)))e-7")
	cpu_guest_nice=$(printf %.2f "$((1000000000 - (10000000 * (100 * cpu_guest_nice) / cpu_delta)))e-7")
 
	echo "P \"CPU utilization (LXC)\" util=${cpu_usage};80;90;0;100|user=${cpu_user};80;90;0;100|system=${cpu_system};80;90;0;100|nice=${cpu_nice};80;90;0;100|idle=${cpu_idle};;;0;100|io_wait=${cpu_iowait};80;90;0;100|interrupt=${cpu_irq};80;90;0;100|si=${cpu_softirq};80;90;0;100|cpu_util_steal=${cpu_steal};80;90;0;100|cpu_util_guest=${cpu_guest};80;90;0;100|cpu_guest_nice=${cpu_guest_nice};80;90;0;100 Total CPU: ${cpu_usage}%"
	exit 0
  fi

  # Keep this as last for our next read 
  cpu_last=("${cpu_now[@]}") 
  cpu_last_sum=$cpu_sum 
  
  # Wait a second before the next read 
  sleep 1.0
done