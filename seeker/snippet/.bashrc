#date: 2024-04-30T16:53:02Z
#url: https://api.github.com/gists/093b22497f1a959ad02f2375e333bc3a
#owner: https://api.github.com/users/Yttehs-HDX

#
# ~/.bashrc
#

# If not running interactively, don't do anything
[[ $- != *i* ]] && return

alias grep='grep --color=auto'
PS1='[\u@\h \W]\$ '

# neovim
export EDITOR='nvim'

# language
export LANG=en_US.UTF-8
export LANGUAGE=en_US

# thefuck
eval $(thefuck --alias)

# alias
alias ls='exa --color=auto --icons'
alias v='nvim'
alias vi='nvim'
alias vim='nvim'
alias c='clear'
alias t='tmux'
alias h='history | sort -nr | cat'
alias ran='ranger'
alias df='duf'
alias cat='bat'
alias tree='exa --color=auto --icons --git --tree'
alias whats='gh copilot explain'
alias howto='gh copilot suggest'
alias draw='tgpt -img'

# functions
transzh() {
	trans :zh $1
}
transen() {
	trans :en $1
}

askgpt() {
	tgpt --url "" \
	--provider "openai" --model "gpt-4" \
	--key "" $1
}

weather() {
	curl wttr.in/$1
}

frp-start() {
	sudo systemctl start sshd frpc
}

frp-stop() {
	sudo systemctl stop sshd frpc
}

frp-enable() {
	sudo systemctl enable --now sshd frpc
}

frp-disable() {
	sudo systemctl disable --now sshd frpc
}

frp-status() {
	systemctl status sshd frpc
}

# custom exports
export PROJECT=/mnt/External/Develop/Project
export PATH=$PATH:~/.local/bin
export PATH=$PATH:~/.cargo/bin