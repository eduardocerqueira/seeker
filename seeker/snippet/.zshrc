#date: 2022-06-20T17:08:53Z
#url: https://api.github.com/gists/45747ccd8573ab73479a9d435ee89e8e
#owner: https://api.github.com/users/phlawlessDevelopment

# Enable Powerlevel10k instant prompt. Should stay close to the top of ~/.zshrc.
# Initialization code that may require console input (password prompts, [y/n]
# confirmations, etc.) must go above this block; everything else may go below.
if [[ -r "${XDG_CACHE_HOME:-$HOME/.cache}/p10k-instant-prompt-${(%):-%n}.zsh" ]]; then
  source "${XDG_CACHE_HOME:-$HOME/.cache}/p10k-instant-prompt-${(%):-%n}.zsh"
fi

export ZSH="$HOME/.oh-my-zsh"

ZSH_THEME="powerlevel10k/powerlevel10k"

plugins=(autoswitch_virtualenv zsh-autosuggestions tmux git python docker docker-compose)

ZSH_TMUX_AUTOSTART=true

source $ZSH/oh-my-zsh.sh

alias dev="cd ~/dev/"
alias python="python3"
alias djrs="py manage.py runserver"
alias djmm="py manage.py makemigrations"
alias djm="py manage.py migrate"
alias djcs="py manage.py collectstatic"
alias djcsu="py manage.py createsuperuser"
alias djsp="django-admin startproject"
alias djsa="py manage.py startapp"

[ -f ~/.fzf.zsh ] && source ~/.fzf.zsh

