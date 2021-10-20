#date: 2021-10-20T17:02:07Z
#url: https://api.github.com/gists/f252cec4357c14f8a8860eb3bee02623
#owner: https://api.github.com/users/orinocoz

#!/bin/sh

KERIO_BACKUP_DIR="/opt/kerio/mailserver/store/backup"

CONFIG="$(dirname "$0")/hetzner-backup-account-info.txt"

TOTAL_LIMIT="$(expr 100 '*' 1024 '*' 1024 '*' 1024)"  # ..100 GB


####  Functions  ###############################################


Fail() {
    echo "$@" | mail -s "${0##*/} failed" admins
    logger -p user.err -t "${0##*/}" -s -- "$@"
    exit 1
}

Verbose() { test -n "$VERBOSE" && echo "$@"; }

LFTP() { lftp -e "$*; quit" $USER:$PASS@$HOST || Fail "lftp failed"; }

Getfiles_for_backup() {
    # Global vars: assign FILES
    FILES="$(find "$KERIO_BACKUP_DIR/" -type f -name '*.zip' -mtime -1)"
    test -n "$FILES" && return 0
    Verbose "nothing to do."
    exit 0
}

Check_files_are_not_toobig() {
    local SUMSIZE="$(du -b $FILES | awk '{ sum += $1 } END { print sum }')"
    test "$SUMSIZE" -ge "$TOTAL_LIMIT" &&
       Fail "New files are too big for backup: limit = $TOTAL_LIMIT, files size = $SUMSIZE"
}

Check_file_isnot_toobig() {
    # Global vars: read FILE and TOTAL_LIMIT, assign SIZE
    SIZE="$(stat -c '%s' "$FILE")"
    test "$SIZE" -ge "$TOTAL_LIMIT" &&
       Fail "New file $FILE is too big for backup: limit = $TOTAL_LIMIT, filesize = $SIZE"
}

Need_free() {
    # Global vars: read TOTAL_LIMIT, SIZE
    local USED="$(LFTP cls -s --block-size=1 | awk '{ sum += $1 } END { print sum }')"
    test -n "$USED" || Fail "LFTP failed, cannot get usage summary."

    local FREE="$(expr "$TOTAL_LIMIT" - "$USED")"
    test "$SIZE" -lt "$FREE" && return 1 # ..dont need to delete old files

    Verbose "size=$SIZE, used=$USED, free=$FREE, need_free=YES"
    return 0
}

Delete_oldest() {
    local OLDEST="$(LFTP cls --sort=date | head -1)"
    Verbose "oldest = $OLDEST"
    test -n "$OLDEST" && LFTP rm "$OLDEST"
}


####  Main  ####################################################


test -s "$CONFIG" || Fail "$CONFIG is missing or empty"
. "$CONFIG"

which lftp >/dev/null 2>&1 || Fail "missing lftp"

Getfiles_for_backup
Check_files_are_not_toobig  # ..needed because Kerio splits large backups to multiple 2GB files

for FILE in $FILES; do

    Check_file_isnot_toobig  # ..actually not needed because overloaded by Check_files_are_not_toobig

    while Need_free; do Delete_oldest; done

    Need_free && Fail "cannot free space on backup storage"

    Verbose "put $FILE"
    LFTP put "$FILE"

done

exit 0

## END ##