#date: 2021-10-05T17:13:02Z
#url: https://api.github.com/gists/1fab14f59aae2d38e0a9752e7f4d1159
#owner: https://api.github.com/users/davigurgel

if type brew &>/dev/null; then
  HOMEBREW_PREFIX="$(brew --prefix)"
  if [[ -r "${HOMEBREW_PREFIX}/etc/profile.d/bash_completion.sh" ]]; then
    source "${HOMEBREW_PREFIX}/etc/profile.d/bash_completion.sh"
  else
    for COMPLETION in "${HOMEBREW_PREFIX}/etc/bash_completion.d/"*; do
      [[ -r "$COMPLETION" ]] && source "$COMPLETION"
    done
  fi
fi

GIT_PS1_SHOWDIRTYSTATE=true                  # *+ # Unstaged (*) and staged (+) changes
GIT_PS1_SHOWSTASHSTATE=true                  # $  # If something is stashed, then a '$' will be shown
GIT_PS1_SHOWUNTRACKEDFILES=true              # %  # If there're untracked files, then a '%' will be shown
GIT_PS1_SHOWUPSTREAM="auto verbose name"     # u  # See the difference between HEAD and its upstream
GIT_PS1_SHOWCOLORHINTS=true                  #    # A colored hint about the current dirty state. Available only when using __git_ps1 for PROMPT_COMMAND or precmd
export EDITOR=vim

source ~/.git-completion.bash                # curl https://raw.githubusercontent.com/git/git/master/contrib/completion/git-completion.bash -o ~/.git-completion.bash
source ~/.git-prompt.sh                      # curl -o ~/.git-prompt.sh \https://raw.githubusercontent.com/git/git/master/contrib/completion/git-prompt.sh

PS1="\[\033[01;30m\]\t\[\033[00m\] \[\033[01;32m\][\w\[\033[01;33m\]\$(__git_ps1 ' (%s)')\[\033[01;32m\]]\$\[\033[00m\] "

alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'

alias ga="git add"
alias gb="git branch"
alias gc="git commit"
alias gca="git commit --amend"
alias gcan="git commit --amend --no-edit"
alias gcm="git commit . -m"
alias gco="git checkout"
alias gcod="gco develop"
alias gcobd="gco baymax/develop"
alias gd="clear && git diff"
alias gdev="gco -b dev || (gb -D dev ; gco -b dev)"
alias gds="gd --staged"
alias gdt="git diff-tree --no-commit-id --name-only -r"
alias gf="git fetch upstream"
alias gfo="git fetch origin"
alias gfod="git fetch origin develop:develop"
alias gfr="gm && gf && gr && gp"
alias gh="~/.git_ps1_help.sh"
alias gl="clear && echo '' && git log -n30 --graph --abbrev-commit --decorate --format=format:'%C(auto)%h%C(reset) %C(white)%s%C(reset) %C(auto)%d%C(reset) %C(dim white)- %an <%ae> (%ad)%C(reset)'"
alias glm=" echo \" >>> Veja gsh\" && echo && git log -1 --pretty=%B && echo \" >>> Veja gsh\""
alias gm="git checkout master"
alias gmr="git fetch $1 merge-requests/$2/head:mr-$1-$2 && git checkout mr-$1-$2"
alias gp="git push"
alias gpf="gp --force"
alias gpr="git pull --rebase"
alias gr="git rebase upstream/master"
alias grod="git rebase origin/develop"
alias grc="git rebase --continue"
alias grs="git rebase --skip"
alias gro="git rebase origin/master"
alias gs="git status"
alias gsh="git show --stat"
alias ios="xcrun simctl list devices"
alias metro="sudo lsof -i :8081"
alias watchman="watchman watch-del-all"
alias android="scrcpy"
alias rndoctor="npx @react-native-community/cli doctor"

[[ -s "$HOME/.rvm/scripts/rvm" ]] && source "$HOME/.rvm/scripts/rvm" # Load RVM into a shell session *as a function*