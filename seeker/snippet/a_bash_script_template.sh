#date: 2023-09-01T16:54:57Z
#url: https://api.github.com/gists/0fd477fb681573f56b49d60915302a0e
#owner: https://api.github.com/users/rieramos

#!/usr/bin/env bash
#
# A b3bp (BASH3 Boilerplate) fork, and a basic bash script template :p
#
# For usage:
#
#	./a_bash_script_template.sh --[FLAGS/OPTIONS] [ARGS]
#
# For (short) changelog:
#
# TO-DOs/Upcoming (Checked are unreleased)
# - [ ] TO-DO.
# - [x] ~~TO-DO~~.
#
# vN.N.N (Last release)
# - Lorem Ipsum.
# - Added Lorem Ipsum.
# - Lorem Ipsum fixed.
#
# For license:
#
# Author: John Doe <user@example.com>
#
# or
#
# The MIT License (MIT)
# Copyright (c) 2013 John Doe and contributors
# You are not obligated to bundle the LICENSE file with your <script_reference> (Name, acronym, etc. Optional) projects as long
# as you leave these references intact in the header comments of your source files.
#
# (Change, add, remove comments/placeholders in function of your script)

[[ "${BASH_SOURCE[0]}" != "${0}" ]] && __tmp_source_idx=1

export __dir="$(cd "$(dirname -- "$(readlink -f "${BASH_SOURCE[${__tmp_source_idx:-0}]}")")" && pwd)"

source "${__dir}/b3bp.sh" #If script is for gist.github, and want visibility, just rename helper file as zzz.
                          #Or make this file hide/put some "a" at the beginning of name.
cd "${__dir}"

main() {
  :
}

if [[ "${BASH_SOURCE[0]}" = "${0}" ]];
then
	main "${@}"
fi
