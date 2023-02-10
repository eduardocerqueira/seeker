#date: 2023-02-10T16:48:00Z
#url: https://api.github.com/gists/43afece541d33ae0dced35decc6aabea
#owner: https://api.github.com/users/MissKittin

#!/bin/sh

### before you start having fun, set it up

# Set user name and email id (required)
GIT_USER_NAME='yourgithubusername'
GITHUB_USER_EMAIL_ID='someNumbersFromStartOfGithubEmail'

# Set email and repo url (required)
GIT_USER_EMAIL="${GITHUB_USER_EMAIL_ID}+${GIT_USER_NAME}@users.noreply.github.com"
GIT_ORIGIN_URL="git@github.com:${GIT_USER_NAME}"

# If GIT_TOOLBOX_GITHUB_GISTS is set, change repo url (see case ${1} clone)
[ ! "${GIT_TOOLBOX_GITHUB_GISTS}" = '' ] && \
	GIT_ORIGIN_URL='git@gist.github.com'

# SSH key - paste id_ed25519 key below (required if ssh used)
SSH_PRIV_KEY=$(cat << EOF
PASTE KEY HERE FROM -----BEGIN OPENSSH PRIVATE KEY----- TO -----END OPENSSH PRIVATE KEY-----
EOF
)

# SSH key - paste id_ed25519.pub key below (required if ssh used)
SSH_PUB_KEY=$(cat << EOF
PASTE KEY HERE
EOF
)

# Add SSH additional options below
SSH_ADDITIONAL_OPTS=''
#SSH_ADDITIONAL_OPTS=' -o "StrictHostKeyChecking=no" -o "UserKnownHostsFile=/dev/null"'

# GPG key - paste gpg-private.key below (required if signing used)
GPG_PRIV_KEY=$(cat << EOF
PASTE KEY HERE FROM -----BEGIN PGP PRIVATE KEY BLOCK----- TO -----END PGP PRIVATE KEY BLOCK-----
EOF
)

# GPG key - paste gpg-public.key below (optional but recommended)
GPG_PUB_KEY=$(cat << EOF
PASTE KEY HERE FROM -----BEGIN PGP PUBLIC KEY BLOCK----- TO -----END PGP PUBLIC KEY BLOCK-----
EOF
)

# GPG owner trust - paste gpg-ownertrust.txt below (required if signing used)
GPG_OWNERTRUST=$(cat << EOF
PASTE TXT FILE HERE
EOF
)

### End of config

SSH_TEMP_DIR=''
[ ! "${0##*/}" = 'git-toolbox.sh-ssh' ] && GIT_SSH_COMMAND=''
unpack_ssh_key()
{
	[ ! "${SSH_TEMP_DIR}" = '' ] && return 1
	SSH_TEMP_DIR=$(mktemp -d)

	echo "${SSH_PRIV_KEY}" > "${SSH_TEMP_DIR}/id_ed25519"
	echo "${SSH_PUB_KEY}" > "${SSH_TEMP_DIR}/id_ed25519.pub"

	chmod 600 "${SSH_TEMP_DIR}/id_ed25519"
	chmod 600 "${SSH_TEMP_DIR}/id_ed25519.pub"

	GIT_SSH_COMMAND='ssh -i '"${SSH_TEMP_DIR}"'/id_ed25519 -o IdentitiesOnly=yes'"${SSH_ADDITIONAL_OPTS}"
	export 'GIT_SSH_COMMAND'

	GIT_TOOLBOX_LEGACY_SSH_COMMAND='ssh -i '"${SSH_TEMP_DIR}"'/id_ed25519 -o IdentitiesOnly=yes'"${SSH_ADDITIONAL_OPTS}"
	export 'GIT_TOOLBOX_LEGACY_SSH_COMMAND'
}
remove_ssh_key()
{
	[ "${SSH_TEMP_DIR}" = '' ] && return 1
	rm -r "${SSH_TEMP_DIR}"
	SSH_TEMP_DIR=''

	GIT_SSH_COMMAND=''
	export 'GIT_SSH_COMMAND'

	GIT_TOOLBOX_LEGACY_SSH_COMMAND=''
	export 'GIT_TOOLBOX_LEGACY_SSH_COMMAND'
}

GPG_TEMP_DIR=''
GNUPGHOME=''
unpack_gpg_key()
{
	[ ! "${GPG_TEMP_DIR}" = '' ] && return 1
	GPG_TEMP_DIR=$(mktemp -d)
	chmod 700 "${GPG_TEMP_DIR}"

	if [ ! "${1}" = 'generate' ]; then
		echo "${GPG_PRIV_KEY}" > "${GPG_TEMP_DIR}/gpg-private.key"
		echo "${GPG_PUB_KEY}" > "${GPG_TEMP_DIR}/gpg-public.key"
		echo "${GPG_OWNERTRUST}" > "${GPG_TEMP_DIR}/gpg-ownertrust.txt"

		chmod 600 "${GPG_TEMP_DIR}/gpg-private.key"
		chmod 600 "${GPG_TEMP_DIR}/gpg-public.key"
		chmod 600 "${GPG_TEMP_DIR}/gpg-ownertrust.txt"
	fi

	GNUPGHOME="${GPG_TEMP_DIR}"
	export 'GNUPGHOME'

	if [ ! "${1}" = 'generate' ]; then
		gpg --import "${GPG_TEMP_DIR}/gpg-private.key"
		gpg --import-ownertrust "${GPG_TEMP_DIR}/gpg-ownertrust.txt"
	fi
}
remove_gpg_key()
{
	[ "${GPG_TEMP_DIR}" = '' ] && return 1
	rm -r "${GPG_TEMP_DIR}"
	GPG_TEMP_DIR=''

	GNUPGHOME=''
	export 'GNUPGHOME'
}

print_S1()
{
	echo -n "${1}"
}

if [ -e "$(dirname "${0}")/.git-toolbox.sh-ssh" ]; then
	if [ "${0##*/}" = '.git-toolbox.sh-ssh' ]; then
		if [ "${GIT_TOOLBOX_LEGACY_SSH_COMMAND}" = '' ]; then
			echo "GIT_TOOLBOX_LEGACY_SSH_COMMAND not exported"
			exit 1
		fi

		${GIT_TOOLBOX_LEGACY_SSH_COMMAND} ${@}
		exit "$?"
	fi

	#echo 'Legacy mode enabled'
	GIT_SSH="$(dirname "$(realpath "${0}")")/.git-toolbox.sh-ssh"
	export 'GIT_SSH'
fi

EXIT_CODE='0'

case "${1}" in
	'init')
		if [ "${2}" = '' ]; then
			echo 'init repo-name'
			exit 1
		fi

		mkdir "${2}" || exit 1

		git init -b master

		#git config --local core.autocrlf false
		#git config --local gui.encoding utf-8
		git config --local user.email "${GIT_USER_EMAIL}"
		git config --local user.name "${GIT_USER_NAME}"
	;;
	'clone')
		if [ "${2}" = '' ]; then
			echo 'clone repo-name'
			exit 1
		fi
		if [ -e "${2}" ]; then
			echo "${2} already exists"
			exit 1
		fi

		unpack_ssh_key
		if [ "${GIT_TOOLBOX_GITHUB_GISTS}" = '' ]; then
			git clone "${GIT_ORIGIN_URL}/${2}.git"
		else
			git clone "${GIT_ORIGIN_URL}:${2}.git"
		fi
		EXIT_CODE="$?"
		remove_ssh_key

		cd "${2}" > /dev/null 2>&1 || exit 1
		git config --local user.email "${GIT_USER_EMAIL}"
		git config --local user.name "${GIT_USER_NAME}"
	;;
	'auto-commit')
		git add -A
		git commit -a -m "${0##*/} auto-commit"
		"${0}" autogc
	;;
	'signed-commit')
		unpack_gpg_key
		git -c "user.signingkey=${GIT_USER_EMAIL}" commit -S ${2} ${3} ${4} ${5} ${6} ${7} ${8} ${9}
		EXIT_CODE="$?"
		remove_gpg_key

		"${0}" autogc
	;;
	'signed-auto-commit')
		exec "${0}" signed-commit -a -m "${0##*/} signed-auto-commit"
	;;
	'signed-tag')
		unpack_gpg_key
		git -c "user.signingkey=${GIT_USER_EMAIL}" tag -s ${2} ${3} ${4} ${5} ${6} ${7} ${8} ${9}
		EXIT_CODE="$?"
		remove_gpg_key

		"${0}" autogc
	;;
	'push')
		if [ "${3}" = '' ]; then
			echo 'push origin-name branch-or-tag-name'
			exit 1
		fi

		unpack_ssh_key
		git push "${2}" "${3}"
		EXIT_CODE="$?"
		remove_ssh_key

		"${0}" autogc
	;;
	'amend')
		exec git commit --amend -a --no-edit
	;;
	'undo-changes')
		exec git reset --hard HEAD
	;;
	'undo-commit')
		exec git reset --soft HEAD~1
	;;
	'reset-branch')
		if [ "${3}" = '' ]; then
			echo 'reset-branch branch-name commit-message'
			exit 1
		fi

		git checkout --orphan sdifgysdhsdg
		git add .
		git add -A
		git commit -a -m "${3}"
		git branch -D "${2}"
		git branch -m sdifgysdhsdg "${2}"

		"${0}" autogc
	;;
	'gc')
		git fsck --unreachable
		git reflog expire --expire=0 --all
		git prune
		#git repack -a -d -l
		#git gc --aggressive
		git gc
	;;
	'autogc')
		loose_objects="$(print_S1 $(git count-objects))"

		if [ "${loose_objects}" -ge '100' ]; then
			echo "Loose objects: ${loose_objects}"
			exec "${0}" 'gc'
		else
			echo "gc: not yet (${loose_objects}/100)"
		fi
	;;
	'generate-ssh-keys')
		exec ssh-keygen -t ed25519 -C "${GIT_USER_EMAIL}"
	;;
	'generate-gpg-keys')
		unpack_gpg_key 'generate'

		GPG_TTY=$(tty)
		export 'GPG_TTY'

		echo 'Y' | gpg --quick-generate-key "${GIT_USER_NAME} <${GIT_USER_EMAIL}>"
		gpg --armor --export "${GIT_USER_EMAIL}" > './gpg-public.key'
		gpg --armor --export-secret-keys "${GIT_USER_EMAIL}" > './gpg-private.key'
		gpg --export-ownertrust > './gpg-ownertrust.txt'

		GPG_TTY=''
		export 'GPG_TTY'

		remove_gpg_key
	;;
	'bash')
		if [ "${2}" = '--ssh' ] || [ "${3}" = '--ssh' ]; then
			unpack_ssh_key
		fi
		if [ "${2}" = '--gpg' ] || [ "${3}" = '--gpg' ]; then
			unpack_gpg_key
		fi

		bash
		EXIT_CODE="$?"

		if [ "${2}" = '--gpg' ] || [ "${3}" = '--gpg' ]; then
			remove_gpg_key
		fi
		if [ "${2}" = '--ssh' ] || [ "${3}" = '--ssh' ]; then
			remove_ssh_key
		fi
	;;
	'enable-legacy')
		if [ -e "$(dirname ${0})/.git-toolbox.sh-ssh" ]; then
			echo 'Legacy mode is already enabled'
			exit 1
		fi

		exec ln -s "./${0##*/}" "$(dirname ${0})/.git-toolbox.sh-ssh"
	;;
	'disable-legacy')
		if [ ! -e "$(dirname ${0})/.git-toolbox.sh-ssh" ]; then
			echo 'Legacy mode is already disabled'
			exit 1
		fi

		exec rm "$(dirname ${0})/.git-toolbox.sh-ssh"
	;;
	'gist')
		GIT_TOOLBOX_GITHUB_GISTS='true'
		export 'GIT_TOOLBOX_GITHUB_GISTS'

		exec "${0}" ${2} ${3} ${4} ${5} ${6} ${7} ${8} ${9}
	;;
	*)
		echo " ${0##*/} init repo-name"
		echo " ${0##*/} clone repo-name"
		echo " ${0##*/} auto-commit"
		echo '  automatically add commit message'
		echo " ${0##*/} signed-commit git-params"
		echo '  do signed commit'
		echo " ${0##*/} signed-auto-commit"
		echo '  automatically add commit message and sign it'
		echo " ${0##*/} signed-tag git-params"
		echo '  do signed tag'
		echo " ${0##*/} push origin-name branch-or-tag-name"
		echo " ${0##*/} amend"
		echo " ${0##*/} undo-changes"
		echo " ${0##*/} undo-commit"
		echo " ${0##*/} reset-branch branch-name commit-message"
		echo '  remove all history'
		echo " ${0##*/} gc"
		echo " ${0##*/} autogc"
		echo " ${0##*/} generate-ssh-keys"
		echo " ${0##*/} generate-gpg-keys"
		echo " ${0##*/} bash [--ssh] [--gpg]"
		echo '  [unpack ssh or/and gpg keys and] open shell'
		if [ -e "$(dirname ${0})/.git-toolbox.sh-ssh" ]; then
			echo " ${0##*/} disable-legacy"
			echo '  removes an .git-toolbox.sh-ssh link from the script directory'
			echo '  and uses the GIT_SSH_COMMAND variable'
		else
			echo " ${0##*/} enable-legacy"
			echo '  creates an .git-toolbox.sh-ssh link in the script directory'
			echo '  and uses the older GIT_SSH variable instead of GIT_SSH_COMMAND'
		fi
		echo " ${0##*/} gist arg1 arg2 arg3"
		echo '  set source to gist.github and reexec this script'
		exit 1
	;;
esac

exit "${EXIT_CODE}"
