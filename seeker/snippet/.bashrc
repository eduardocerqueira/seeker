#date: 2022-09-16T17:23:53Z
#url: https://api.github.com/gists/48e42f27631f69015abe4c2a6d28ee59
#owner: https://api.github.com/users/furgelisherpa

#
# ~/.bashrc
#

# If not running interactively, don't do anything
[[ $- != *i* ]] && return

alias ls='ls --color=auto'
PS1='\e[0;31m[\e[0m\e[0;33m\u\e[0m\e[0;32m@\e[0m\e[0;034m\h\e[0m  \e[0;95m\W\e[0m\e[0;31m]\e[0m# '