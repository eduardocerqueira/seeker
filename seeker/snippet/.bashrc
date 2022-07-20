#date: 2022-07-20T17:07:33Z
#url: https://api.github.com/gists/f1f7d58d1147c10831f1547813b4a1db
#owner: https://api.github.com/users/markizano

export DATEFORMAT='%F/%R:%S';
export HISTCONTROL=ignoreboth
export HISTFILE=$HOME/.bash_history
export HISTFILESIZE=-1
export HISTSIZE=81920
export HISTTIMEFORMAT="$DATEFORMAT "
export HISTLOG=~/.bash_history.d
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:.:~/bin
export PROMPT_COMMAND='echo -en "\n\e[36m$?\e[0m; "; history -a; histlog "$(history 1)"'

# Place subsequent `histlog` command in ~/bin/histlog and it should take care of the rest.
