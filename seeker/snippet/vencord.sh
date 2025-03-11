#date: 2025-03-11T16:50:26Z
#url: https://api.github.com/gists/913419499aa5b58cb33217e196402f2c
#owner: https://api.github.com/users/TheSainEyereg

#!/usr/bin/env bash
set -e

say_error() {
	# {
	# 	GTK_THEME="Breeze-Dark" zenity --error --text="$1" 2>/dev/null
	# } || {
	# 	echo -e "\033[0;31m$1\033[0m"
	# }
	echo -e "\033[0;31m$1\033[0m"
}

if [ "$EUID" -eq 0 ]; then
	say_error "Do not run this install script as root"
	exit 1
fi

outfile=$(mktemp)
trap 'rm -f "$outfile"' EXIT

echo "Downloading Vencord installer"

curl -sS https://github.com/Vendicated/VencordInstaller/releases/latest/download/VencordInstallerCli-Linux \
	--output "$outfile" \
	--location

chmod +x "$outfile"

if command -v sudo >/dev/null; then
	echo "Running with sudo"
	sudo "$outfile" "$@"
elif command -v doas >/dev/null; then
	echo "Running with doas"
	doas "$outfile" "$@"
else
	say_error "No sudo or doas found"
fi