#date: 2022-05-02T17:08:53Z
#url: https://api.github.com/gists/98411498a8f94f648f8edd6fecd80a1b
#owner: https://api.github.com/users/SteveontheMoon

#!/bin/bash

function pacman_backup(){
	tar -cjf pacman_database.tar.bz2 /var/lib/pacman/local
}

function pacman_install(){
	tar -xjvf pacman_database.tar.bz2	
}

while getopts "bi" opts
do
		case $opts in
				b) pacman_backup;;
				i) pacman_install;;
				*) echo "Pass -b to make backup or -i to install backup" && exit 1;;
		esac
done

