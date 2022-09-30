#date: 2022-09-30T17:13:12Z
#url: https://api.github.com/gists/263ed3fd35756ab0355087ffd7245b46
#owner: https://api.github.com/users/bng44270

##############################################
#
# iptcli - Auto-complete iptables command line interface
#
# Usage:
#
#    1. Load script:  source /path/to/iptcli.inc.sh
#    2. Run iptcli function
#
# iptcli syntax:
#
#    iptcli <table> <action> [<chain>] [<option>]
#
#    <action> = append|check|delete|insert|replace|list|flush|zero|newchain|delchain
#    <option> = interface|protocol|source|destination|match|destport|jump
#
##############################################

iptcli() {
	iptables $(sed '
		s/^/-t /;
		s/ append / -A /;
		s/ check / -C /;
		s/ delete / -D /;
		s/ insert / -I /;
		s/ replace / -R /;
		s/ list/ -L /;
		s/ flush / -F /;
		s/ zero / -Z /;
		s/ newchain / -N /;
		s/ delchain / -X /;
		s/ interface / -i /;
		s/ protocol / -p /;
		s/ source / -s /;
		s/ destination / -d /;
		s/ match / -m /;
		s/ destport / --dport /;
		s/ jump / -j / 
	' <<< "$@")
}

_str_begin_array() {
	NEWAR=()
	SRCH="$(cat -)"
	for EL in $@; do
		if [ -n "$(grep "^$SRCH" <<< "$EL")" ]; then
			NEWAR+=($EL)
		fi
	done
	echo ${NEWAR[@]}
}

_in_array() {
	SRCH="$(cat -)"
	if [ "$1" == "$SRCH" ]; then
		return 0
	else
		shift
		AR="$@"
		if [ -z "$AR" ]; then
			return 1
		else
			_in_array $AR <<< "$SRCH"
		fi
	fi
}

_iptcli_complete() {
	ROOTCMD=(iptcli)
	TABLES=(nat filter mangle raw security)
	ACTIONS=(append check delete insert replace list flush zero newchain delchain)
	OPTIONS=(interface protocol source destination match destport jump)
	[[ -n "${COMP_WORDS[1]}" ]] && CHAINS=($(iptables-save -t ${COMP_WORDS[1]} 2> /dev/null | awk '/^:/ { printf("%s ",gensub(/^:/,"","g",$1)); }'))
	
	LASTCMD="${COMP_WORDS[$COMP_CWORD-1]}"
		
	_in_array ${ROOTCMD[@]} <<< "$LASTCMD"
	if [ $? -eq 0  ]; then
		COMPREPLY=($(_str_begin_array ${TABLES[@]} <<< "${COMP_WORDS[$COMP_CWORD]}"))
		return 0
	fi
	
	_in_array ${TABLES[@]} <<< "$LASTCMD"
	if [ $? -eq 0 ]; then
		COMPREPLY=($(_str_begin_array ${ACTIONS[@]} <<< "${COMP_WORDS[$COMP_CWORD]}"))
		return 0
	fi
	
	_in_array ${ACTIONS[@]} <<< "$LASTCMD"
	if [ $? -eq 0 ]; then
		COMPREPLY=($(_str_begin_array ${CHAINS[@]} <<< "${COMP_WORDS[$COMP_CWORD]}"))
		return 0
	fi
	
	_in_array ${OPTIONS[@]} <<< "$LASTCMD"
	if [ $? -eq 0 ]; then
		return 0
	fi
	
	COMPREPLY=($(_str_begin_array ${OPTIONS[@]} <<< "${COMP_WORDS[$COMP_CWORD]}"))
	return 0
}

complete -F _iptcli_complete iptcli