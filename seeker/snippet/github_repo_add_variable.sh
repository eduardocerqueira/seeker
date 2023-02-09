#date: 2023-02-09T16:52:52Z
#url: https://api.github.com/gists/8d9e7f0356388b538bc508f284bf87e4
#owner: https://api.github.com/users/sebastian-garaycoechea-simplisafe

#!/bin/bash

if [ -z $ACTION_RUNNER_GROUP ]; then
	read -p 'Please enter ACTION_RUNNER_GROUP value: ' ACTION_RUNNER_GROUP
fi

if [ -z $TOKEN ]; then
	echo "Please declare GitHub TOKEN variable to pull $ACTION_RUNNER_GROUP repo list.."
	exit 1
fi

REPOS=$(curl -ss -H "Authorization: "**********"
	-H 'Accept: application/vnd.github.v3.raw' \
	https://raw.githubusercontent.com/simplisafe/tf-github-actions-infra/main/teams/${ACTION_RUNNER_GROUP}/runnergroups/standard/repos.txt)

echo "Grabbing repository list for $ACTION_RUNNER_GROUP's team."
echo "Runner group: standard"

for repo in $REPOS; do
	status_code=$(curl \
  	-X POST \
	-s -o /dev/null -w "%{http_code}" \
  	-H "Accept: application/vnd.github+json" \
  	-H "Authorization: "**********"
  	-H "X-GitHub-Api-Version: 2022-11-28" \
  	https://api.github.com/repos/simplisafe/$repo/actions/variables \
	-d '{"name":"ACTION_RUNNER_GROUP","value":"'"$ACTION_RUNNER_GROUP"'"}')
	
	if [[ $status_code -eq 201 ]]; then
		echo "ACTION_RUNNER_GROUP with value $ACTION_RUNNER_GROUP has been added to $repo repository."
	fi
done

	fi
done
