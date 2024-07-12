#date: 2024-07-12T16:49:09Z
#url: https://api.github.com/gists/ef44453f2da72d2f55492b7d4c955d8d
#owner: https://api.github.com/users/LeXofLeviafan

#!/bin/bash
# Usage: `bukuserver` starts up the server, `bukuserver --stop` sends TERM to the already running server
# On startup, the user is offered to choose or create a new DB (cancel both to exit)
# After the server stops (via CTRL+C or `bukuserver --stop`), the choice is given again
# NOTE: requires Zenity to work

: "${BUKUSERVER=$HOME/Work/buku/}"              # path to executable, or directory to run from source
: "${VENV:=$HOME/.local/share/bukuserver/venv}" # alternatively, can be set to $BUKUSERVER/venv
BUKU_CONFIG_DIR=~/.local/share/buku

# specify Web-UI settings here
export BUKUSERVER_THEME=slate
export BUKUSERVER_DISABLE_FAVICON=false
export BUKUSERVER_OPEN_IN_NEW_TAB=true


function _settermtitle { echo -en "\033]2;$1\007"; }  # changes terminal title

function _select-db {
	FILES=( )
	while read FILE; do
		[ -e "$FILE" ] && FILES+=( "$(basename "${FILE%.db}")" )
	done < <(ls -1 "$BUKU_CONFIG_DIR"/{,.}*.db 2>/dev/null | sort)
	FILE=
	if [ ${#FILES[@]} != 0 ]; then
		FILE=`zenity --list --title="Choose DB" --text="(or click Cancel to create new DB)" --column="Name" -- "${FILES[@]}"`
		[ "$FILE" ] && echo "$BUKU_CONFIG_DIR/$FILE.db" && return
	fi
	while true; do
		FILE=`zenity --entry --title="Create new DB?" --text="DB name (cannot contain '/'):" --entry-text="bookmarks"`
		! [ "$FILE" ] && echo "No name given, qutting" >&2 && return
		[[ "$FILE" == *'/'* ]] && zenity --error --text="DB name cannot contain '/'!" && continue
		[ -e "$BUKU_CONFIG_DIR/$FILE.db" ] && ! zenity --question --text="'$FILE' exists already. Open anyway?" && continue
		echo "$BUKU_CONFIG_DIR/$FILE.db"
		return
	done
}


if [ "$1" == '--stop' ]; then
	PID=`ps -afu "$USER" | grep '/python[^ ]* .*/bukuserver run$' | awk '{print $2}'`
	[ "$PID" ] && kill "$PID"
	exit
fi

_settermtitle 'bukuserver'

if [ -d "$BUKUSERVER" ]; then
	cd "$BUKUSERVER"
	python -m venv "$VENV"
	. "$VENV/bin/activate"
	pip install .[server]
	BUKUSERVER='bukuserver'
fi

export BUKUSERVER_DB_FILE=`_select-db`
while [ "$BUKUSERVER_DB_FILE" ]; do
	"$BUKUSERVER" run
  export BUKUSERVER_DB_FILE=`_select-db`
done
