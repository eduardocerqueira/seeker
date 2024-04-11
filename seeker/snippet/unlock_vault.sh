#date: 2024-04-11T16:49:05Z
#url: https://api.github.com/gists/ae3fe1f68f76483fe8babd902b737b14
#owner: https://api.github.com/users/pythoninthegrass

#!/usr/bin/env bash

# $USER
[[ -n $(logname >/dev/null 2>&1) ]] && logged_in_user=$(logname) || logged_in_user=$(whoami)

# check os
if [[ ! $(uname) = "Darwin" ]]; then
	echo "This script is for macOS only."
	exit 1
fi

add_keychain_password() {
	local mypass
	read -s -p "Enter ansible vault password: "**********"
	security add-generic-password \
		-a "$logged_in_user" \
		-s "ansible-vault" \
		-w "$mypass" \
		-T "/usr/bin/security"
}

check_app_password() {
	app_password= "**********"
		-a "$logged_in_user" \
		-s "ansible-vault" -w 2>&1 >/dev/null)
	rc=$(echo $?)
	if [[ $rc -ne 0 ]]; then
		echo "No password found in keychain. "
		add_keychain_password
	fi
}

print_vault_password() {
	security find-generic-password \
		-a $logged_in_user \
		-s ansible-vault -w
}

main() {
	check_app_password
	print_vault_password
}
main

exit 0
d
}
main

exit 0
