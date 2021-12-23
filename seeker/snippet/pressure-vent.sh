#date: 2021-12-23T16:59:25Z
#url: https://api.github.com/gists/cd54c8d8bf3803599d3b33cc56e6fbc0
#owner: https://api.github.com/users/evg-zhabotinsky

#!/bin/bash
# Steam "Pressure Vessel" runtime eliminator.
#
# Valve ships various pre-packaged runtime environments for games to use.
# They are cool, in a way: Most games "just work" with them.
# However, such convenience comes at the cost of performance and sometimes reliability.
#
# Normally, I disable any and all such runtimes and install all the required dependencies myself.
# However, running Windows games using Proton enforces use of a runtime.
# That runtime is shipped in a "Pressure Vessel" container, which is more isolated.
# The worst part is, user-supplied LD_PRELOAD and LD_LIBRARY_PATH are ignored due to that.
# That means no primusrun/pvkrun/whatever for Windows games, which sucks.
#
# This little script's purpose is to cut "Pressure Vessel" out of game's command line.
# Be warned that that gives you the ability *and responsibility* to manage the dependencies.
# Place pressure-vent.sh after all " -- " in game's launch options, if any.
# Placing e.g. "xterm -e" at the start of launch options is an easy way to see logs.
# Examples:
#   primusrun ~/pressure-vent.sh %command%
#   xterm -e /full/path/to/pressure-vent.sh pvkrun %command%


# Functions
cmdprn() {
	printf '%s:\n%q' "$1" "$2"
	printf ' %q' "${@:3}"
	printf '\n\n'
}
err() {
	printf '%s\nPress Enter to quit.\n' "$1"
	read
	exit 1
}

# Debug
cmdprn "Original command line" "$@"
printf "LD_PRELOAD:\n%q\n\n" "$LD_PRELOAD"
printf "LD_LIBRARY_PATH:\n%q\n\n" "$LD_LIBRARY_PATH"

# Find Pressure Vessel arguments (between two first "--")
((left=1))
while [[ left -le $# && "${!left}" != "--" ]]; do
	((left++))
done
((right=left+1))
while [[ right -le $# && "${!right}" != "--" ]]; do
	((right++))
done
[[ right -gt $# ]] && err 'Error processing command line.'
# Cut them out
set -- "${@:1:left}" "${@:right+1}"
cmdprn "Processed command line" "$@"  # Debug

"$@" || err "Game terminated with code $?."

# Comment out if it actually waits for Enter and you don't like it
printf 'Finished without error. Press Enter to quit.\n' && read