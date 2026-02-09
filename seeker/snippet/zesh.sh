#date: 2026-02-09T17:37:07Z
#url: https://api.github.com/gists/097947f38422bb65f327bca72fe87ad7
#owner: https://api.github.com/users/ewok

#!/usr/bin/env bash

ZELLIJ_SWITCH="zellij pipe --plugin https://github.com/mostafaqanbaryan/zellij-switch/releases/download/0.2.1/zellij-switch.wasm"

list_zellij_sessions() {
	zellij list-sessions --short 2>/dev/null | awk '{print $1}'
}

list_zoxide_dirs() {
	zoxide query -l 2>/dev/null
}

list_combined() {
	list_zellij_sessions
	list_zoxide_dirs
}

export -f list_zellij_sessions
export -f list_zoxide_dirs
export -f list_combined

is_existing_session() {
	local name="$1"
	zellij list-sessions --short 2>/dev/null | awk '{print $1}' | grep -qx "$name"
}

is_directory() {
	local path="$1"
	[[ -e "$path" && -d "$path" ]]
}

derive_session_name() {
	local path="$1"
	local parent_name=$(basename "$(dirname "$path")")
	local base_name=$(basename "$path")
	echo "${parent_name}__${base_name}"
}

SESS=$(list_combined | grep -vE "^(opencode|claude)$" | fzf \
	--border-label ' zellij session manager ' --prompt '⚡  ' \
	--header '  ^a all ^t zellij ^x zoxide ^d delete session ^e zoxide erase ^f find' \
	--bind 'tab:down,btab:up' \
	--bind 'ctrl-a:change-prompt(⚡  )+reload(list_combined | grep -vE "^(opencode|claude)$")' \
	--bind "ctrl-t:change-prompt(🪟  )+reload(list_zellij_sessions)" \
	--bind "ctrl-x:change-prompt(📁  )+reload(list_zoxide_dirs)" \
	--bind 'ctrl-f:change-prompt(🔎  )+reload(fd -L -H -d 5 -t d -E .Trash -E .git -E .cache . ~)' \
	--bind "ctrl-d:execute(zellij delete-session {})+change-prompt(⚡  )+reload(list_zellij_sessions)" \
	--bind "ctrl-e:execute(zoxide remove {})+change-prompt(⚡  )+reload(list_zoxide_dirs)")

if [[ -n $SESS ]]; then
	SESS="${SESS/#\~/$HOME}"
	if [[ -e "$SESS" ]]; then
		if [[ -L "$SESS" ]]; then
			SESS=$(readlink -f "$SESS")
		else
			SESS=$(realpath "$SESS")
		fi
	fi

	if [[ -n "$ZELLIJ" ]]; then
		if is_existing_session "$SESS"; then
			$ZELLIJ_SWITCH -- "--session $SESS --layout default"
		elif is_directory "$SESS"; then
			session_name=$(derive_session_name "$SESS")
			$ZELLIJ_SWITCH -- "--session $session_name --cwd $SESS --layout default"
		else
			$ZELLIJ_SWITCH -- "--session $SESS --layout default"
		fi
	else
		if is_directory "$SESS"; then
			session_name=$(derive_session_name "$SESS")
			cd "$SESS" && zellij attach --create "$session_name" --layout default
		else
			zellij attach --create "$SESS" --layout default
		fi
	fi
fi
