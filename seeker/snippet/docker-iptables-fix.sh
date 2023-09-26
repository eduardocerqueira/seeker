#date: 2023-09-26T17:01:51Z
#url: https://api.github.com/gists/87ebd72a897b22be33a43de45e57399f
#owner: https://api.github.com/users/almottier

#!/bin/bash
currentAttempt=0
totalAttempts=10
delay=15

while [ $currentAttempt -lt $totalAttempts ]
do
	currentAttempt=$(( $currentAttempt + 1 ))
	
	echo "Attempt $currentAttempt of $totalAttempts..."
	
	result=$(iptables-save)

	if [[ $result =~ "-A DOCKER -i docker0 -j RETURN" ]]; then
		echo "Docker rules found! Modifying..."
		
		iptables -t nat -A PREROUTING -m addrtype --dst-type LOCAL -j DOCKER
		iptables -t nat -A PREROUTING -m addrtype --dst-type LOCAL ! --dst 127.0.0.0/8 -j DOCKER
		
		echo "Done!"
		
		break
	fi
	
	echo "Docker rules not found! Sleeping for $delay seconds..."
	
	sleep $delay
done
