#date: 2026-02-17T17:26:40Z
#url: https://api.github.com/gists/13d06b4e8e1a83b157ded90f2c5f2355
#owner: https://api.github.com/users/ghostrydr

function killport() { 
    if [ -z $1 ]; then
        echo "Usage: killport <port>"
        return
    fi
    lsof -t -i tcp:$1 | xargs kill -9
}
function whatport()  {
    if [ -z $1 ]; then
        echo "Usage: whatport <port>"
        return
    fi
    lsof -i tcp:$1 | grep 'LISTEN' | awk '{print $2}' | xargs ps -fp | awk '{ print $9 }'
}

# Git
alias status='git status'
alias fetch='git fetch'
alias pull='git pull'
alias push='git push'
alias stash='git stash'
alias pop='git stash pop'
alias abort='git reset --merge'
alias mm='git merge master'
alias checkout='git checkout $@'
alias branch='git checkout -b $@'