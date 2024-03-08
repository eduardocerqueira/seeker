#date: 2024-03-08T17:01:23Z
#url: https://api.github.com/gists/c1364d07af0060e4687bbc46c63b6b0a
#owner: https://api.github.com/users/Jcpetrucci

#!/bin/bash
query='autodiscover.wip.company.com.'
declare -A currentAnswer=()
declare -A lastAnswer=()
while :; do
        #hr;
        date '+%F %T.%N %Z';
        for resolver in geodns1-int.company.com. geodns2-int.company.com.; do
                varResolver=${resolver%%-*}
                currentAnswer[$varResolver]="$(dig @${resolver} ${query} +short;)"
                echo ${currentAnswer[$varResolver]}
                if [[ "${currentAnswer[$varResolver]}" != "${lastAnswer[$varResolver]}" ]]; then
                        printf '%s\n' 'This answer differs from the last!'
                fi
                lastAnswer[$varResolver]="${currentAnswer[$varResolver]}"
        done
        sleep 10;
done