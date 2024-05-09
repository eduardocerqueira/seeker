#date: 2024-05-09T17:04:09Z
#url: https://api.github.com/gists/a13d8e1d4708b12da48f6640aec62b1a
#owner: https://api.github.com/users/j3h1

#
# ~/.bashrc
#

# If not running interactively, don't do anything
[[ $- != *i* ]] && return

set_ps1() {
  # https://gist.github.com/fnky/458719343aabd01cfb17a3a4f7296797#256-colors
  # ^ Use the 256-colors IDs
	local accent_color_1='228'
	local accent_color_2='59'
	local text_color_1='59'
	local text_color_2='228'
	local dirchar_color='25'

	local pwd2=$(echo "$PWD" | sed -e "s:$HOME:~:" -e "s:\([^/]\)/:\1$(printf ' \356\202\261 '):g")
	PS1="\n\342\224\214\342\224\200 \[\e[38;5;$(echo $accent_color_1)m\]\356\202\262\[\e[0m\e[48;5;$(echo $accent_color_1)m\e[38;5;$(echo $text_color_1)m\] \u@\h \[\e[0m\e[48;5;$(echo $accent_color_2)m\e[38;5;$(echo $accent_color_1)m\]\356\202\260\[\e[38;5;$(echo $text_color_2)m\] $pwd2 \[\e[0m\e[38;5;$(echo $accent_color_2)m\]\356\202\260\n\[\e[38;2;121;79;153m\]$\[\e[0m\] "
}

PROMPT_COMMAND="set_ps1"

clear