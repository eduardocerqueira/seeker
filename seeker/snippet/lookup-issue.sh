#date: 2024-04-10T16:57:28Z
#url: https://api.github.com/gists/1b6eda4b2a254293a943698166c07090
#owner: https://api.github.com/users/attiss

#!/bin/bash

nameservers=(a1-207.akam.net. a20-66.akam.net. a14-66.akam.net. a6-64.akam.net. a7-67.akam.net. a5-64.akam.net.)


success=0
failure=0

for n in $(seq 0 100); do
	for ns in ${nameservers[@]}; do
		dig a0730a114e51268787150-6b64a6ccc9c596bf59a86625d8fa2202-c000.us-east.satellite.appdomain.cloud @${ns} +retry=0 +time=1 > /dev/null
		if [[ $? -eq 0 ]]; then
			echo -en "success "
			success=$((success+1))
		else
			echo -en "failure(${ns}) "
			failure=$((failure+1))
		fi
	done
done

echo $success
echo $failure
