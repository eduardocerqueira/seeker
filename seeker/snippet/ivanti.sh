#date: 2021-08-31T03:14:00Z
#url: https://api.github.com/gists/017e41ab87ece7da615cb47cab90c108
#owner: https://api.github.com/users/dbr787

#!/bin/bash

DATE=$(date)
DAD_JOKE=$(curl https://icanhazdadjoke.com/ -H "Accept: application/json")
JSON='{"fact_evaluated_for":"'"$1"'","fact_evaluated_on":"'"$HOSTNAME"'","fact_evaluated_at":"'"$DATE"'"}'
echo $JSON $DAD_JOKE | jq -s add
