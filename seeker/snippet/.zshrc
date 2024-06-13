#date: 2024-06-13T16:54:36Z
#url: https://api.github.com/gists/e9a1183aa1536edc900026b1a9c6a3d0
#owner: https://api.github.com/users/piprees

export ZSH="$HOME/.oh-my-zsh"
export CLICOLOR=1
export ZSH_THEME="arrow"
export CASE_SENSITIVE="true"

autoload -Uz compinit && compinit
plugins=(gitfast yarn npm node python pip rake scala vscode macos rails ruby docker aws)
source $ZSH/oh-my-zsh.sh
unset LESS;

export PROMPT='$ %{$reset_color%}'
export RPROMPT=''

alias fortune='/c/bin/fortune/fortune.exe'

alias rename='mv';
alias cls='clear';
alias logdog='git log -6 --graph --abbrev-commit --decorate=no --format=format:"%C(02)%>|(16)%h%C(reset) %<(16,trunc)%cr %<(26,trunc)%cl %<(90,mtrunc)%s"';
alias logadog='git log -20 --graph --abbrev-commit --decorate=no --format=format:"%C(02)%>|(16)%h%C(reset) %<(16,trunc)%cr %<(26,trunc)%cl %<(90,mtrunc)%s"';
alias ls='LS -L -N --color --group-directories-first --format=across --file-type';
alias pls='ls'
alias als='LS -A -L -N --color --group-directories-first --format=across --file-type';
alias ll='als';

function _LS(){
    clear;
    if [ -e "CHANGELOG.md" ];
    then echo ;
    else
        echo --------- ;
        fortune ;
        echo ;
    fi;
    if [ -d ".git" ] || [ -d "../.git" ] || [ -d "../../.git" ] || [ -d "../../../.git" ];
    then
        echo ;
        echo commits ------ ;
        logdog;
        echo ;
        echo branches ------ ;
        echo -n \> latest;
        git branch;
        echo ;
        echo status ------ ;
        git status;
        echo ;
    fi;
    echo ${PWD##/*/}/ ------ ;
    pls;
    echo ;
};

function _CD(){
   builtin cd "$@";
   _LS;
}

alias l='_LS';
alias s='_LS';
alias c='_LS';
alias d='_LS';
alias start='yarn start';

alias projects="cd /d/projects";
alias npm-default='npm config set registry https://registry.npmjs.org && yarn config set registry https://registry.yarnpkg.com'


clear;
# ls;