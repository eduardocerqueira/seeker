#date: 2022-02-07T16:50:54Z
#url: https://api.github.com/gists/cddf67b498c96ed6ffca19e7301be793
#owner: https://api.github.com/users/jessebutryn

#!/usr/bin/env bash

runs=$1

passorfail () {
	local _thing=$1
	sleep 2
	if [[ $_thing == pass ]]; then
		return 0
	else
		return 1
	fi
}

for ((i=1;i<=runs;i++)); do
	if (((RANDOM%10)+1>5)); then
		thing=pass
	else
		thing=fail
	fi
	passorfail "$thing" &
	pids+=($!)
done

for pid in "${pids[@]}"; do
	if wait "$pid"; then
		((passed++))
	else
		((failed++))
	fi
done

printf 'Passed: %d\nFailed: %d\n' "$passed" "$failed"
