#date: 2022-02-14T16:58:32Z
#url: https://api.github.com/gists/a55e751dae6a0afa1ad47b941cf0e995
#owner: https://api.github.com/users/megapod

#!/bin/bash

main() {
        br
	echo "login script show some information to start with."
	sep
        br
	echo "sensors"
	sep
        br
	sensors
	br
	echo "Hard drive temp:"
	sep
	hddtemp
        br
	echo "top cpu consumers:"
	sep
        br
	ps -eo pcpu,pid,user,args | sort -k 1 -r | head -10
	br
	echo "top memory consumers"
	sep
	ps aux --sort -rss | head -10
	br
	echo "hard drive space"
	sep
	df -h
	br
	echo "LAST 50 LOGINS"
	sep
	last | head -50
	br
	echo "show all users last login"
	sep
	lastlog | more	
	br
	echo "changed files at home folder last 24 hours:"
	br
	sep
	find . -maxdepth 1 -mtime -1 -type f -exec ls -l {} \;
	br
	echo "get the last 50 installed packages: "
	br
	sep
	rpm -qa --last | head -50

}

br() {
  echo ""
}

sep() {
  echo "--------------------------------------------------"
}
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
