#date: 2023-02-23T16:58:11Z
#url: https://api.github.com/gists/a4bb376ca1acdf39d3f0eb0d3e4fe023
#owner: https://api.github.com/users/svandragt

#!/usr/bin/env bash
#{{{ Bash settings
# abort on nonzero exitstatus
set -o errexit
# abort on unbound variable
set -o nounset
# don't hide errors within pipes
set -o pipefail
#}}}

on_payslip() {
	# Rename with year and month, and file away into the Payslips folder
	year_month=$(date +%Y-%m)
	f=$2
	mv "$f" "~/Payslips/${f%.*}-$year_month.${f##*.}"
}

_main() { 
	source "$SCRIPT_DIR/$SCRIPT_NAME"
	# When ~/Downloads/PaySlip.pdf is created call on_payslip
	TARGETDIR=~/Downloads
	cd $TARGETDIR
 	inoticoming --foreground --initialsearch --logfile .folderactions.log --pid-file .folderactions.pid . --suffix PaySlip.pdf --stderr-to-log -- "$SCRIPT_DIR/$SCRIPT_NAME" on_payslip "{}" \; 
 }

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
	#{{{ Variables
	IFS=$'\t\n'   # Split on newlines and tabs (but not on spaces)
	SCRIPT_NAME=$(basename "${0}")
	SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
	readonly SCRIPT_NAME SCRIPT_DIR
	#}}
	if (( $# == 0 )); then
		_main
	else
		$1 "$@"
	fi
fi
