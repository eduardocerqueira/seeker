#date: 2023-08-07T17:07:43Z
#url: https://api.github.com/gists/16cd0c8c5281f7b87dd261adf79d6cf1
#owner: https://api.github.com/users/karnzx

_print(){
	if [[ -z "$1" ]]; then
		local funcname=${FUNCNAME[0]}
		echo -e "Usage:"
		echo -e "$funcname texts (bold|faint|italic|underline) (red|green|yellow|white)"
		echo -e "Try:"
		echo -e "$funcname hello"
		echo -e "$funcname hello bold"
		echo -e "$funcname hello bold red"
	fi
	local style_n
	local color
	local texts="$1"
	case "$2" in
			bold)
				style_n=1 ;;
			faint)
				style_n=2 ;;
			italic)
				style_n=3 ;;
			underline)
				style_n=4 ;;
			*)
				style_n=0 ;;
	esac
	case "$3" in
		red)
			color=31 ;;
		green)
			color=32 ;;
		yellow)
			color=33 ;;
		* | white)
			color=97 ;;
	esac
	echo -e "\e[${style_n};${color}m${texts}\e[0m"
}

_print "hello world" underline green