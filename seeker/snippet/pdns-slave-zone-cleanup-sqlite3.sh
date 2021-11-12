#date: 2021-11-12T17:10:47Z
#url: https://api.github.com/gists/b3271e6192c7de383a1f0a5c70f6aa19
#owner: https://api.github.com/users/capalon

#!/bin/bash
#### Config ################################

DATABASE="/var/lib/powerdns/pdns.sqlite3"

DEBUG="no"

#### End of Config #########################

REQUIRED_COMMANDS="
sqlite3
host
grep
awk
tail
"

# print debug messages to STDERR
function debug {
        if [ "${DEBUG}" == "yes" ] ; then
                echo "DEBUG: $@" >&2
        fi
}

for CMD in ${REQUIRED_COMMANDS} ; do
        CMDNAME=`echo ${CMD} | awk '{print toupper($1) }' | sed -e s@"-"@""@g`
        export $(eval "echo ${CMDNAME}")=`which ${CMD} 2>/dev/null`
        if [ -z "${!CMDNAME}" ] ; then
                debug "Command: ${CMD} not found!"
                exit 1
        else
                debug "Found command $(echo $CMDNAME) in ${!CMDNAME}"
        fi
done

SQLCMD="${SQLITE3} ${DATABASE}"
debug "Sql command: ${SQLCMD}"

check() {
	debug "Check: ${1} ${2}"
	AUTH=`${HOST} -t SOA ${2}. ${1} | ${TAIL} -n1 | ${GREP} "not found"`
	if [ "${AUTH}" == "Host ${2} not found: 5(REFUSED)" ]; then
		debug "Server ${1} has no zone for ${2}"
		DOMAIN_ID=`${SQLCMD} "SELECT id FROM domains WHERE name = '${2}' AND type = 'SLAVE' AND master LIKE '%${1}%' LIMIT 1;"`
		if [ "${DOMAIN_ID}" != "" ]; then
			debug "Removing zone ${2}"
			${SQLCMD} "DELETE FROM records WHERE domain_id = '${DOMAIN_ID}';"
			${SQLCMD} "DELETE FROM domains WHERE id = '${DOMAIN_ID}';"
		fi
	fi
}

MASTERS=(`${SQLCMD} "SELECT DISTINCT ip FROM supermasters;"`)
for m in "${MASTERS[@]}"; do
	debug "Master: ${m}"
	NAMES=(`${SQLCMD} "SELECT name FROM domains WHERE type = 'SLAVE' AND master LIKE '%${m}%';"`)
	for d in "${NAMES[@]}"; do
		check ${m} ${d}
	done
done
