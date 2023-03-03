#date: 2023-03-03T16:43:51Z
#url: https://api.github.com/gists/2a49c96d56bf7e0d25c345338f5e73fd
#owner: https://api.github.com/users/SacrificeGhuleh

export ZSH="$HOME/.oh-my-zsh"

ZSH_THEME="zsh2000"
export ZSH_2000_DISABLE_RVM='true'


ENABLE_CORRECTION="true"
zstyle ':omz:update' mode auto      # update automatically without asking

plugins=(
    git
    colored-man-pages
    zsh-autosuggestions
    zsh-syntax-highlighting
    history-substring-search
    tmux
)

ZSH_TMUX_AUTOSTART="true"
ZSH_TMUX_AUTOCONNECT="false"

PATH=$PATH:~/.local/bin

source $ZSH/oh-my-zsh.sh

alias cls='clear'
alias clean='clear'

alias ls='ls --color=auto'

setopt INC_APPEND_HISTORY   # Write to the history file immediately, not when the shell exits.
setopt SHARE_HISTORY        # Share history between all sessions.
setopt HIST_SAVE_NO_DUPS    # Don't write duplicate entries in the history file.

PROMPT_COMMAND="history -a;$PROMPT_COMMAND"
alias tmux='tmux -2'