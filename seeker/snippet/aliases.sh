#date: 2021-09-22T17:11:36Z
#url: https://api.github.com/gists/4edee52e85820fcf9807fc1e30ae9dca
#owner: https://api.github.com/users/tbro-evi

alias netstat-nlt="lsof -nP -i4TCP | grep LISTEN"
alias curb='git branch | grep \* | sed "s/^\* //g"'
alias gpo='git push origin `curb`'
alias set-branch='git branch --set-upstream-to=origin/`curb` `curb`'
alias force-push='git push --force-with-lease origin `curb`'
alias bgroup='git branch | grep'
alias nvml='. "$NVM_DIR/nvm.sh"'
alias rvml='source "$HOME/.rvm/scripts/rvm"'
alias use-rvm="source ~/.rvm/scripts/rvm"
alias bashrl='source ~/.bash_profile'
alias alias-list='cat ~/.bash_profile | egrep "^alias"'
alias lsf="ls -F | grep /"
alias cdproj="cd ~/Projects"
alias reset-test="RAILS_ENV=test rails db:drop && RAILS_ENV=test rails db:create"
alias reset-dev="RAILS_ENV=development rails db:drop && RAILS_ENV=development rails db:create"
alias migrate-test="RAILS_ENV=test rails db:migrate"
alias check-sql="psql -h 127.0.0.1 -p 5432 -U lion -l"
alias stage-cli="aws ecs execute-command --profile lion-stage --cluster evi-lion-stage-lion-cluster --task 3f18d54a0b7e48ceabd181a3c1ad536d --container backend-lion-stage --command \"/bin/sh\" --interactive"
alias dev-cli="heroku run --app activityx-lion-backend-dev /bin/sh"
alias qa-cli="heroku run --app activityx-lion-backend-qa /bin/sh"