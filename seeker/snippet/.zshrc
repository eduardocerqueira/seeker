#date: 2022-05-06T17:13:46Z
#url: https://api.github.com/gists/fa4174fb872f8b76536a115300231bd1
#owner: https://api.github.com/users/UltimateNova1203

# Export paths for shell to use
export HOMEBREW_EDITOR='/user/bin/nano'
export PATH='/usr/local/opt/llvm/bin:~/Library/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin'
export LDFLAGS='-L/usr/local/opt/llvm/lib'
export CPPFLAGS='-I/usr/local/opt/llvm/include'

# History
HISTFILE='~/.zsh_history'
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

autoload -U colors && colors
PROMPT='%F{green}%n@%m%{%}%f:%F{cyan}%1~%f %% '