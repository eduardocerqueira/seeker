#date: 2023-02-17T16:43:03Z
#url: https://api.github.com/gists/3ec26d504f60c84b49c2a7d770dae135
#owner: https://api.github.com/users/mtfurlan

#!/bin/bash
set -euo pipefail
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

set +e
read -r -d '' backups <<'EOF'
[
    { "src": "/home/mark/sync" },
    { "src": "/home/mark/projects" },
    { "src": "/home/mark/beatSaber" },
    { "src": "/mnt/media" },
    {
      "src": "/home/mark/backup",
      "dest": "home-backup"
    },
    {
        "src": "/scratch/dte-outage-tracking",
        "dest": "dte-outage-history"
    }
]
EOF

# shellcheck disable=SC2120
help () {
    # if arguments, print them
    [ $# == 0 ] || echo "$*"

    cat <<EOF
Usage: $(basename "${BASH_SOURCE[0]}") [OPTION]...
  backup hardcoded paths to script dir
Available options:
  -h, --help       display this help and exit
  -n, --dry-run    print commands don't execute
EOF

    # if args, exit 1 else exit 0
    [ $# == 0 ] || exit 1
    exit 0
}

msg() {
    echo >&2 -e "${1-}"
}

die() {
    local msg=$1
    local code=${2-1} # default exit status 1
    msg "$msg"
    exit "$code"
}

# getopt short options go together, long options have commas
TEMP=$(getopt -o hn --long help,dry-run -n "$0" -- "$@")
#shellcheck disable=SC2181
if [ $? != 0 ] ; then
    die "something wrong with getopt"
fi
eval set -- "$TEMP"

dryRun=false
while true ; do
    case "$1" in
        -h|--help) help; exit 0; shift ;;
        -n|--dry-run) dryRun=true ; shift ;;
        --) shift ; break ;;
        *) die "issue parsing args, unexpected argument '$0'!" ;;
    esac
done

if [ "$dryRun" = true ]; then
    msg "doing a dry run"
fi

rsyncCommand="rsync -av --delete"

for row in $(echo "${backups}" | jq -r '.[] | @base64'); do
    _jq() {
        echo "${row}" | base64 --decode | jq -r "${1}"
    }

    src="$(_jq '.src')"
    dest="$(_jq '.dest')"
    if [[ "$dest" == "null" ]]; then
        dest="$DIR/$(basename "$src")"
    fi
    fullCommand="$rsyncCommand $src/ $dest/"
    if [ "$dryRun" = true ]; then
        echo "$fullCommand"
    else
        $fullCommand
    fi
done