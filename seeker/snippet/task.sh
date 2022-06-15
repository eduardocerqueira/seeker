#date: 2022-06-15T17:04:24Z
#url: https://api.github.com/gists/ceaafed8cbf34e31fd098fff10848f88
#owner: https://api.github.com/users/TheDevtop

#!/bin/bash

# Prog: Tasklist in bash and SQLite
# Auth: Thijs Haker

# Makes life much easier
TASKFILE="./task.db"
VALUE=$2

# Prints usage
function fn_usage {
    printf 'task [add/del/list/init] [text/id]\n'
    return
}

# Creates database
function fn_init {
    touch $TASKFILE
    sqlite3 $TASKFILE 'PRAGMA foreign_keys = off;
    BEGIN TRANSACTION;
    CREATE TABLE tasktable (id INTEGER PRIMARY KEY ASC AUTOINCREMENT UNIQUE NOT NULL, text TEXT UNIQUE, date TEXT NOT NULL);
    COMMIT TRANSACTION;
    PRAGMA foreign_keys = on;'
    return
}

# Lists tasks
function fn_list {
    RESULT=`sqlite3 $TASKFILE 'SELECT * FROM tasktable;'`
    printf "$RESULT\n"
    return
}

# Adds task
function fn_add {
    DATE=`date +%d/%m/%Y`
    sqlite3 $TASKFILE "INSERT INTO tasktable(text,date) VALUES(\"$VALUE\",\"$DATE\");"
    return
}

# Deletes task
function fn_delete {
    sqlite3 $TASKFILE "DELETE FROM tasktable WHERE id=$VALUE;"
    return
}

case "$1" in
    init)
    fn_init ;;
    list)
    fn_list ;;
    add)
    fn_add ;;
    del)
    fn_delete ;;
    *)
    fn_usage ;;
esac
exit 0
