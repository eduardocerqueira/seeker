#date: 2022-05-18T17:15:53Z
#url: https://api.github.com/gists/1d41f687c8a485a4304631273a8b5b53
#owner: https://api.github.com/users/vletty

#!/bin/bash

session="session"

tmux new-session -d -s "${session}"

window=0
tmux send-keys -t ${session}:${window} "vi /tmp/test.txt" C-m

window=1
tmux new-window -t ${session}:${window} -n "ssh"
tmux send-keys -t ${session}:${window} "ssh server.example.com" C-m

window=2
tmux new-window -t ${session}:${window} -n "split"
tmux split-window -t ${session}:${window} -h
tmux split-window -t ${session}:${window} -v
