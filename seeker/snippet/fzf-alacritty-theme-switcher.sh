#date: 2022-08-26T16:43:13Z
#url: https://api.github.com/gists/f416bc0fdb3eeadb66d4454f76bbbc70
#owner: https://api.github.com/users/chrisgrieser

# switch alacritty color scheme. requires `fzf` and `alacritty-colorscheme` (pip package).
# save alacritty themes in ~/.config/alacritty/colors, download from https://github.com/eendroroy/alacritty-theme
function t(){
	local selected colordemo
	local alacritty_color_schemes=~/.config/alacritty/colors
	local orignal=$(alacritty-colorscheme status | cut -d. -f1)
	local input="$*"
	read -r -d '' colordemo << EOM
\033[1;30mblack  \033[0m  \033[1;40mblack\033[0m
\033[1;31mred    \033[0m  \033[1;41mred\033[0m
\033[1;32mgreen  \033[0m  \033[1;42mgreen\033[0m
\033[1;33myellow \033[0m  \033[1;43m\033[1;30myellow\033[0m
\033[1;34mblue   \033[0m  \033[1;44mblue\033[0m
\033[1;35mmagenta\033[0m  \033[1;45mmagenta\033[0m
\033[1;36mcyan   \033[0m  \033[1;46m\033[1;30mcyan\033[0m
\033[1;37mwhite  \033[0m  \033[1;47m\033[1;30mwhite\033[0m
EOM

	# --preview-window=0 results in a hidden preview window, with the preview
	# command still taking effect. together, they create a "live-switch" effect
	selected=$(ls "$alacritty_color_schemes" | cut -d. -f1 | fzf \
					-0 -1 \
					--query="$input" \
					--expect=ctrl-y \
					--cycle \
					--ansi \
					--height=8 \
					--layout=reverse \
					--info=inline \
					--preview-window="right,70%,border-left" \
					--preview="alacritty-colorscheme apply {}.yaml || alacritty-colorscheme apply {}.yml ;echo \"\n$colordemo\"" \
	         )

	# re-apply original color scheme when aborting
	if [[ -z "$selected" ]] ; then
		alacritty-colorscheme apply "$orignal.yaml" || alacritty-colorscheme apply "$orignal.yml"
		return 0
	fi

	key_pressed=$(echo "$selected" | head -n1)
	selected=$(echo "$selected" | tail -n+2)

	if [[ "$key_pressed" == "ctrl-y" ]] ; then
		cat "$selected" | pbcopy
	else
		alacritty-colorscheme apply "$selected.yaml" || alacritty-colorscheme apply "$selected.yml"
	fi
}