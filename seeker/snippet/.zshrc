#date: 2022-10-28T17:13:56Z
#url: https://api.github.com/gists/814280eea153bc4532ed4e4e1742697e
#owner: https://api.github.com/users/andrewiadevaia

# Created by newuser for 5.8.1
eval "$(starship init zsh)"
source ~/.zsh/zsh-autosuggestions/zsh-autosuggestions.zsh
source ~/.zsh/zsh-syntax-highlighting/zsh-syntax-highlighting.zsh

HISTFILE=~/.zsh_history
HISTSIZE=10000
SAVEHIST=10000
setopt appendhistory

ZSH_HIGHLIGHT_STYLES[suffix-alias]=fg=blue,underline
ZSH_HIGHLIGHT_STYLES[precommand]=fg=blue,underline
ZSH_HIGHLIGHT_STYLES[arg0]=fg=yellow

alias cat='bat --paging=never'
alias ls="exa -l --icons --sort=name --group-directories-first"
alias ll="ls -a"

bindkey '^[[1;5D' backward-word
bindkey '^[[1;5C' forward-word