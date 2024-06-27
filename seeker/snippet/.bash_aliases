#date: 2024-06-27T16:51:35Z
#url: https://api.github.com/gists/3deb5257d6a17876944a86cfe1c1d54e
#owner: https://api.github.com/users/Voltra

#!/bin/bash

## Git
alias git-add="git add ."
alias git-commit="git commit -a -m"
alias git-push="git push origin master"
alias git-pull="git pull origin master"

alias git-s="git status"
alias git-d="git diff"
alias git-crlf="git config --global core.safecrlf"
alias git-rm="git rm -r --cached"
alias git-ac="git-add && git-commit"
alias git-amend="git commit --amend"

alias git-b="git rev-parse --abbrev-ref HEAD"
alias gpush='git push origin $(git-b)'
alias gpull='git pull origin $(git-b)'

alias gf="git flow"
alias gffs="gf feature start"
alias gffp="gf feature publish"
alias gfff="gf feature finish"
alias gfrs="gf release start"
alias gfrf="gf release finish"
alias gfhs="gf hotfix start"
alias gfhf="gf hotfix finish"

alias gitArchive='git archive --prefix ${PWD##*/}/ HEAD -o ../${PWD##*/}.zip'
alias gitPatch="git apply --unsafe-paths --ignore-space-change"

## Specialization
alias clipboard="clip.exe" # adapt this
alias explorer="explorer.exe" # adapt this
alias gui="explorer.exe ."
alias openGui="explorer.exe"
alias cc="gcc"
alias rename="mv"
alias makeSymLink="ln -s"
alias chmodsh="chmod +x *.sh"
alias artisan="php artisan"
alias osquery="osqueryi"
alias jq='\jq -C'
alias adonis="node ace"
alias tree='\tree -I node_modules'
alias ngrokServe="ngrok http --host-header=rewrite"
alias npmr="npm run"
#alias vim="nvim"

## Scripts
scripts=(tac split_by)
for script in "${scripts[@]}"; do
	alias $script="~/.bash/scripts/$script.sh"
done
chmod +x ~/.bash/scripts/*.sh


## CUSTOM
alias vue="winpty vue.cmd"
alias node="winpty node"
alias npm="winpty npm.cmd"
alias msdl="winpty msdl.cmd"
alias make="winpty make"
alias pip="python3 -m pip"
alias npmRelease="gpush && git switch master && gpush --tags && npmr deploy:docs && npm publish && git switch dev"
