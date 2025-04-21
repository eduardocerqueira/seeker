#date: 2025-04-21T16:39:45Z
#url: https://api.github.com/gists/12c54801bd214293bb87c35061898f11
#owner: https://api.github.com/users/zumikkebe

#! /bin/sh

Usage ()
{
	cat << EOF >&2
usage	:	$(basename "${0}") <file|dir>
EOF
	exit 1
}

if [ ! -z "${1}" ] ; then
	pkgd_file="${1}"
else
	if [[ ! -t 0 ]] ; then
		pkgd_file=$(filepanel -d /boot/system/apps -k fds)
	else
		Usage
	fi
fi

if [ -e "${pkgd_file}" ] ; then
	pkg_src=$(catattr -d SYS:PACKAGE_FILE "${pkgd_file}" 2>/dev/null)
	if [[ "${pkg_src}" ]] ; then
		open /boot/system/packages/${pkg_src}
	else
		cat << EOF >&2
error : ${pkgd_file} does not belong to a package
EOF
		exit 1
	fi
else
	Usage
fi
