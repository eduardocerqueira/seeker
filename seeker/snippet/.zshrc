#date: 2023-04-25T16:44:24Z
#url: https://api.github.com/gists/5a32f63da03638f88351ae1419d8ca85
#owner: https://api.github.com/users/Kabongosalomon

# Load the 'git' plugin
source /opt/homebrew/share/zsh-autosuggestions/zsh-autosuggestions.zsh

function parse_git_branch() {
    git branch 2> /dev/null | sed -n -e 's/^\* \(.*\)/[\1]/p'
}

COLOR_DEF=$'%f'
COLOR_USR=$'%F{32}'
COLOR_DIR=$'%F{149}'
COLOR_GIT=$'%F{253}'
NEW_LINE=$'\n'
setopt PROMPT_SUBST
export PROMPT='${COLOR_USR}%n ${COLOR_DIR}%~ ${COLOR_GIT}$(parse_git_branch)${COLOR_DEF}${NEW_LINE}%% '


# Aliases for daily work with Docker (you may need to add sudo into "docker-compose") this need to be added in .bashrc
alias dsa='docker stop $(docker ps -a -q)' # Docker stop all
alias dc='docker-compose'
alias dcu='dc up'
alias dcuf='dc up --force-recreate'
alias dcd='dc down'
alias dcs='dc start'
alias dcr='dc restart'
alias dcst='dc stop'
alias dcb='dc build' # Docker compose build
alias dil='docker image ls' # Docker image list
alias dcl='docker container ls' # Docker container list

alias kaggle="~/.local/bin/kaggle"

autoload -Uz compinit && compinit