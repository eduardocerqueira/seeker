#date: 2022-06-03T17:14:45Z
#url: https://api.github.com/gists/579b4003932cc50d251b0fcb99e6eb01
#owner: https://api.github.com/users/stedolan

while :; do clear; curl -skL https://bar.emf.camp/api/on-tap.json | jq -r 'keys[] as $k | ("--", ("\($k):"), (.[$k][] | "\(.manufacturer) \(.name) (\(.abv)%)@\(.price)\(if .remaining_pct|tonumber<5 then " (running out)" else "" end)"))' | column -t -s '@'; sleep 10; done