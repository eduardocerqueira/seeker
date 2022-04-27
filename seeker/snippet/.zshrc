#date: 2022-04-27T17:12:37Z
#url: https://api.github.com/gists/6b7f0bd4a5afb5fa50280e24d9371dd5
#owner: https://api.github.com/users/hazeycode

autoload -U compinit colors vcs_info history-search-end
colors
compinit

zstyle ':vcs_info:*' stagedstr '%F{green}●%f '
zstyle ':vcs_info:*' unstagedstr '%F{yellow}●%f '
zstyle ':vcs_info:git:*' check-for-changes true
zstyle ':vcs_info:git:*' formats "%F{blue}%b%f %u%c"

zle -N history-beginning-search-backward-end history-search-end
zle -N history-beginning-search-forward-end history-search-end
bindkey "^[[A" history-beginning-search-backward-end
bindkey "^[[B" history-beginning-search-forward-end
