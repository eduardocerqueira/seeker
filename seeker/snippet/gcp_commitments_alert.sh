#date: 2022-12-26T16:52:33Z
#url: https://api.github.com/gists/f53fb9534824a29532fc011dd5c0af29
#owner: https://api.github.com/users/vadirajks

#!/bin/bash
SlackChannelName=slack-channel
SlackHookURL="https://hooks.slack.com/services/dsdsdsdsds"
MessageBody=
MessageTitle="CRITICAL:/custom-scripts/gcp_commitments_alert.sh ($(hostname)/$(hostname -i))"
for project in prokect1 project2; do
	gcloud compute commitments list --project $project --format="csv(selfLink.scope(regions),name,resources,startTimestamp.date('%Y-%m-%d'),endTimestamp.date('%Y-%m-%d'),type,status)" | sed "s/\"{'type': '//g" | sed "s/', 'amount': '/=/g" | sed "s/'}\",/,/g" | sed "s/'}\";/:/g" | awk -F ',' '{gsub(/[\/].*/, "", $1); print}' OFS="," > /tmp/commitments_${project}.txt || exit 2
	sed -i '/^self_link/d' /tmp/commitments_${project}.txt 2> /dev/null || continue
	COUNT=0
	for region in $(awk -F, '{print $1}' /tmp/commitments_${project}.txt  | sort | uniq); do
		#	EXPIRED
		COUNT=0
		if [ -s  /tmp/commitments_${project}.txt ] && [ $(grep $region /tmp/commitments_${project}.txt 2> /dev/null | grep -c EXPIRED) -gt 0 ]; then
			for i in $(grep EXPIRED /tmp/commitments_${project}.txt | sort | grep ^$region | awk 'NF{NF-=2}1' FS=',' OFS=','); do
				set $(echo $i |sed 's/MEMORY/M/g;s/VCPU/C/g;s/LOCAL_SSD/LS/g'|tr "," "\n")
				days=$(echo $(( ($(date +%s) - $(date --date="$5" +%s) )/(60*60*24) )))
				ALERT_CRITICAL[${COUNT}]=$(echo "[${days} gone]:$2,$3,$5\n") && COUNT=$((COUNT+1)); 
			done
		fi
		if [ "${#ALERT_CRITICAL[@]}" -gt 0 ]; then
			MessageBody=$(echo -e "\`EXPIRED_${region}[M=MEMORY,C=VCPU,LS=LOCAL_SSD]:\`\n$(printf '%s' "\`\`\`${ALERT_CRITICAL[@]}"\`\`\`)")
			[ -n "$MessageBody" ] && bash -x /custom-scripts/postToSlack -t "$MessageTitle" -b "$MessageBody" -c "$SlackChannelName" -u "$SlackHookURL" -s "critical"
		fi
		unset c_details ALERT_CRITICAL days
		COUNT=0

		# ACTIVE
		if [ -s  /tmp/commitments_${project}.txt ] && [ $(grep $region /tmp/commitments_${project}.txt 2> /dev/null | grep -c ACTIVE) -gt 0 ]; then
			for c_details in $(grep ACTIVE /tmp/commitments_${project}.txt | sort| grep ^$region|sed 's/MEMORY/M/g;s/VCPU/C/g;s/LOCAL_SSD/LS/g'); do
				days=$(echo $(( ($(date --date="$(echo $c_details | awk -F, '{print $5}')" +%s) - $(date +%s) )/(60*60*24) )))
				if [ $days -lt 10 ]; then
					set $(echo $c_details | tr "," "\n")
					ALERT_CRITICAL[${COUNT}]=$(echo "[${days} days]:$2,$3,$5\n") && COUNT=$((COUNT+1)); 
				fi
			done
			if [ "${#ALERT_CRITICAL[@]}" -gt 0 ]; then
				MessageTitle="$MessageTitle (Going to Expire)"
				MessageBody=$(echo -e "\`$region[M=MEMORY,C=VCPU,LS=LOCAL_SSD]:\`\n$(printf '%s' "\`\`\`${ALERT_CRITICAL[@]}"\`\`\`)")
				[ -n "$MessageBody" ] && bash -x /custom-scripts/postToSlack -t "$MessageTitle" -b "$MessageBody" -c "$SlackChannelName" -u "$SlackHookURL" -s "critical"
				MessageTitle="CRITICAL:/custom-scripts/gcp_commitments_alert.sh ($(hostname)/$(hostname -i))"
			fi
		fi
		unset c_details ALERT_CRITICAL days
	done
done