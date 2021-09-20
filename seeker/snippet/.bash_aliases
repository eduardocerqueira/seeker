#date: 2021-09-20T16:48:45Z
#url: https://api.github.com/gists/d2b74153fe137a2d0aee0074732163ca
#owner: https://api.github.com/users/tomanistor

alias git-prune='git branch --merged | grep -v "\*" | grep -Ev "(\*|production|master|staging|development)" | xargs -n 1 git branch -d'