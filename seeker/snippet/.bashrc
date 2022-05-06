#date: 2022-05-06T17:20:49Z
#url: https://api.github.com/gists/4427447c9d93a8b31fd850e269233d84
#owner: https://api.github.com/users/UltimateNova1203

# History
HISTFILE='~/.bash_history'
HISTSIZE=1000

# enable color support for la, and aliases
alias grep='grep --color=auto'
alias fgrep='fgrep --color=auto'
alias egrep='egrep --color=auto'
alias ll='ls -la'
alias la='ls -A'
alias l='ls -CF'
alias rm-dm='find ./ -name ".DS_Store" -depth -exec rm []\;'

export CLICOLOR=1
export LSCOLORS=GxFxCxDxBxegedabagaced
export PS1="\[\e[32m\]\u\[\e[m\]\[\e[32m\]@\[\e[m\]\[\e[32m\]\h\[\e[m\]:\[\e[34m\]\W\[\e[m\] \\$ "