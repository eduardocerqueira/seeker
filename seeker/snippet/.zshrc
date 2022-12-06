#date: 2022-12-06T17:06:02Z
#url: https://api.github.com/gists/94441838ef5d34e34ac554392cfd9a70
#owner: https://api.github.com/users/brianmowens

function parse_git_branch() {
    git branch 2> /dev/null | sed -n -e 's/^\* \(.*\)/[\1]/p'
}

setopt PROMPT_SUBST
export PROMPT='%F{grey}%n%f %F{cyan}%~%f %F{green}$(parse_git_branch)%f %F{normal}$%f '