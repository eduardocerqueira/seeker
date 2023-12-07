#date: 2023-12-07T17:00:20Z
#url: https://api.github.com/gists/07b45fb47f01e406ff92ecb73a28f609
#owner: https://api.github.com/users/Cr4zyy

#!/bin/bash
#needs jq

#lxc amp container check
if ! lxc-info AMP | grep -q "State:.*RUNNING"; then
    exit 1
fi


# Make a POST request using curl and capture the output
login_json=$(curl -s -X POST -H "Accept: "**********": application/json" -d '{"username": "USER", "password":"PASSWORD", "token":"", "rememberMe":"False"}' http://127.0.0.1:8080/API/Core/Login)

sessionID=$(echo "$login_json" | jq -r '.sessionID')

#get all instances
instances_json=$(curl -s -X POST -H "Accept: text/javascript" -H "Content-Type: application/json" -d "{\"SESSIONID\":\"$sessionID\"}" http://127.0.0.1:8080/API/ADSModule/GetInstances)

#get player counts
activeUsers=$(echo "$instances_json" | jq -r '.result[].AvailableInstances[] | select(.Metrics != null and .Metrics."Active Users" != null) | .Metrics."Active Users".RawValue // 0')

threshold=1

while read raw_value; do
      if [ "$raw_value" -ge "$threshold" ]; then
        #echo "enable turbo"
        echo "0" > /sys/devices/system/cpu/intel_pstate/no_turbo
      else
        #echo "disable turbo"
        echo "1" > /sys/devices/system/cpu/intel_pstate/no_turbo
      fi
done <<< "$activeUsers"

#logout
curl -s -X POST -H "Accept: text/javascript" -H "Content-Type: application/json" -d "{\"SESSIONID\":\"$sessionID\"}" http://127.0.0.1:8080/API/Core/Logout >/dev/null
re/Logout >/dev/null
