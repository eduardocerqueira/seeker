#date: 2022-11-18T16:55:38Z
#url: https://api.github.com/gists/4a8a9954816388353b73113737230e62
#owner: https://api.github.com/users/bnordli

#!/bin/sh


# === For Windows ===
git config --global user.email "bnordli@gmail.com"
git config --global core.editor 'C:/Program Files (x86)/TextPad 7/TextPad.exe' -m
git config --global core.autocrlf true

# === For unix
git config --global user.email "nordli@exabel.com"
git config --global core.editor "atom -w"
git config --global core.autocrlf input

# === For all ===

git config --global user.name "Børge Nordli"
git config --global rerere.enabled true
git config --global push.default current
git config --global color.diff auto
git config --global color.status auto
git config --global color.branch auto
git config --global color.interactive true
git config --global rebase.autosquash true
git config --global alias.ca "commit --amend"
git config --global alias.co "checkout"
git config --global alias.cob "checkout -b"
git config --global alias.cot "checkout -t"
git config --global alias.f "fetch -p"
git config --global alias.c "commit"
git config --global alias.p "push"
git config --global alias.po "push origin"
git config --global alias.ba "branch -a"
git config --global alias.bd "branch -d"
git config --global alias.bdd "branch -D"
git config --global alias.dc "diff --cached"
git config --global alias.st "status -sb"
git config --global alias.a "add -p"
git config --global alias.plog "log --graph --pretty='format:%C(red)%d%C(reset) %C(yellow)%h%C(reset) %ar %C(green)%aN%C(reset) %s'"
git config --global alias.tlog "log --stat --since='1 Day Ago' --graph --pretty=oneline --abbrev-commit --date=relative"
git config --global alias.lg "log --color --graph --pretty=format:'%Cred%h%Creset -%C(yellow)%d%Creset %s %Cgreen(%cr) %C(bold blue)<%an>%Creset' --abbrev-commit"
git config --global alias.rank "shortlog -sn --no-merges"
git config --global status.submodulesummary 1
git config --global push.recurseSubmodules check

# See https://github.com/magicmonty/bash-git-prompt
# GIT_PROMPT_ONLY_IN_REPO=1
# GIT_PROMPT_THEME=Default_Ubuntu
# source ~/.bash-git-prompt/gitprompt.sh
