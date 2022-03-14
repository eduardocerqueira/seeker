#date: 2022-03-14T17:03:42Z
#url: https://api.github.com/gists/23dc7bea4ce427c50b4dc7ded41e6655
#owner: https://api.github.com/users/alussana

#!/bin/bash

function nospaces {
	for OLDNAME in *
	do
    	NEWNAME=$(echo "${OLDNAME}" | tr -s ' ' '_')
	    if [ "${OLDNAME}" != "${NEWNAME}" ]
	    then
	        mv "${OLDNAME}" "${NEWNAME}"
	    fi
	done
}

function surf_dirs {
	cd $1
	nospaces
	for D in *
	do
		if [ -d "${D}" ]
		then
			surf_dirs $(echo $(pwd)/${D})
			cd ..
		fi
	done
}

surf_dirs $1