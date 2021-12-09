#date: 2021-12-09T16:57:11Z
#url: https://api.github.com/gists/730bfe58717fd50dc3a7735bd3715339
#owner: https://api.github.com/users/markcummins

## Checks the site one time
curl -Is https://www.google.com | grep --color=auto HTTP | awk '$2 != 200 {system("notify-send -t 60000 \"PANIC\"")}'

## Wraps the above command in a 90 second loop
while true; do curl -Is https://www.google.com | grep --color=auto HTTP | awk '$2 != 200 {system("notify-send -t 60000 \"PANIC\"")}' & sleep 90; done