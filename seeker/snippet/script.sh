#date: 2024-04-22T16:54:16Z
#url: https://api.github.com/gists/e29ea06e061c7b9839719f10a335eac3
#owner: https://api.github.com/users/Taiwolawal

#!/bin/bash

env="zettaday"
aws_command="/usr/bin/aws"
sleep_time=60
threshold=80

log_group_name="${env}-HighUtilizationLogs"
metric_namespace="HighUtilization"

# Check if log group exists
log_group_exists=$("$aws_command" logs describe-log-groups --log-group-name-prefix "$log_group_name" 2>/dev/null | grep -c "$log_group_name")

# If log group doesn't exist, create it
if [[ $log_group_exists -eq 0 ]]; then
    echo "Creating log group: $log_group_name"
    "$aws_command" logs create-log-group --log-group-name "$log_group_name"
fi

create_log_stream() {
    local log_stream_name="$1"
    "$aws_command" logs create-log-stream --log-group-name "$log_group_name" --log-stream-name "$log_stream_name" 2>/dev/null
}

while true; do
    cpu_usage=$(top -bn1 | awk 'NR>7{s+=$9} END {print s}')
    memory_usage=$(free | awk 'NR==2{printf "%.2f", $3/$2*100}')

    echo "Checking CPU and memory usage..."

    if (( $(echo "$cpu_usage >= $threshold || $memory_usage >= $threshold" | bc -l) )); then
        echo "High resource usage detected! Processing..."

        ps_output=$(ps -eo user,pid,%cpu,%mem,cmd --sort=-%cpu,%mem --no-headers | head -n 10)

        while IFS= read -r line; do
            user=$(echo "$line" | awk '{print $1}')
            pid=$(echo "$line" | awk '{print $2}')
            cpu=$(echo "$line" | awk '{print $3}')
            mem=$(echo "$line" | awk '{print $4}')
            command=$(echo "$line" | awk '{$1=""; $2=""; $3=""; $4=""; sub(/^ */, "", $0); print $0}')

            cpu_value=$(echo "scale=2; ($cpu_usage / 100) * 100" | bc)

            aws_put_metric_result=$("$aws_command" cloudwatch put-metric-data \
                --namespace "$metric_namespace" \
                --metric-data "{\"MetricName\":\"CPUUsage\",\"Dimensions\":[{\"Name\":\"User\",\"Value\":\"$user\"}],\"Unit\":\"Percent\",\"Value\":$cpu_value,\"Timestamp\":$(date +%s)}")
            aws_exit_status=$?

            if [[ $aws_exit_status -eq 0 ]]; then
                echo "Successfully sent data for user: $user (CPU: $cpu_value%, Memory: $memory_usage%)"
                echo "  (PID: $pid, user: $user, CPU%: $cpu, memory: $mem, command: $command)"
            else
                echo "Error sending data for user: $user (CPU: $cpu_value%, Memory: $memory_usage%). AWS CLI exit status: $aws_exit_status"
            fi
        done <<< "$ps_output"
    else
        echo "Normal resource usage. Sleeping..."
    fi

    sleep $sleep_time
done