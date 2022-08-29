#date: 2022-08-29T17:09:32Z
#url: https://api.github.com/gists/58d13520f3c00d954656de1c8e777faf
#owner: https://api.github.com/users/Victrid

#!/bin/bash

hash git 2>/dev/null || { echo >&2 "Required command 'git' is not installed. ( hmm... why are you using this? ) Aborting."; exit 1; }
hash realpath 2>/dev/null || { echo >&2 "Required command 'realpath' is not installed. Aborting."; exit 1; }
hash pwd 2>/dev/null || { echo >&2 "Required command 'pwd' is not installed. Aborting."; exit 1; }
hash cd 2>/dev/null || { echo >&2 "Required command 'cd' is not installed. Aborting."; exit 1; }
hash echo 2>/dev/null || { echo >&2 "Required command 'echo' is not installed. Aborting."; exit 1; }
hash mv 2>/dev/null || { echo >&2 "Required command 'mv' is not installed. Aborting."; exit 1; }
hash diff 2>/dev/null || { echo >&2 "Required command 'diff' is not installed. Aborting."; exit 1; }
hash diffstat 2>/dev/null || { echo >&2 "Required command 'diffstat' is not installed. Aborting."; exit 1; }
hash tail 2>/dev/null || { echo >&2 "Required command 'tail' is not installed. Aborting."; exit 1; }
hash awk 2>/dev/null || { echo >&2 "Required command 'awk' is not installed. Aborting."; exit 1; }
hash mkdir 2>/dev/null || { echo >&2 "Required command 'mkdir' is not installed. Aborting."; exit 1; }
hash rm 2>/dev/null || { echo >&2 "Required command 'rm' is not installed. Aborting."; exit 1; }

# argument parser generated online by https://argbash.io/generate
# # When called, the process ends.
# Args:
# 	$1: The exit message (print to stderr)
# 	$2: The exit code (default is 1)
# if env var _PRINT_HELP is set to 'yes', the usage is print to stderr (prior to $1)
# Example:
# 	test -f "$_arg_infile" || _PRINT_HELP=yes die "Can't continue, have to supply file as an argument, got '$_arg_infile'" 4
die()
{
	local _ret="${2:-1}"
	test "${_PRINT_HELP:-no}" = yes && print_help >&2
	echo "$1" >&2
	exit "${_ret}"
}


# Function that evaluates whether a value passed to it begins by a character
# that is a short option of an argument the script knows about.
# This is required in order to support getopts-like short options grouping.
begins_with_short_option()
{
	local first_option all_short_options='bvg'
	first_option="${1:0:1}"
	test "$all_short_options" = "${all_short_options/$first_option/}" && return 1 || return 0
}

# THE DEFAULTS INITIALIZATION - POSITIONALS
# The positional args array has to be reset before the parsing, because it may already be defined
# - for example if this script is sourced by an argbash-powered script.
_positionals=()
# THE DEFAULTS INITIALIZATION - OPTIONALS
_arg_branch="master"
_arg_verbose="off"
_arg_generate="off"


# Function that prints general usage of the script.
# This is useful if users asks for it, or if there is an argument parsing error (unexpected / spurious arguments)
# and it makes sense to remind the user how the script is supposed to be called.
print_help()
{
	printf '%s\n' "Git Grafter"
	printf 'Usage: %s [-b|--branch <arg>] [-v|--(no-)verbose] [-g|--(no-)generate] [--help] <original> <leaf>\n' "$0"
	printf '\t%s\n' "<original>: The original git root directory"
	printf '\t%s\n' "<leaf>: The root directory you want to find"
	printf '\t%s\n' "-b, --branch: Set branches (default: 'master')"
	printf '\t%s\n' "-v, --verbose, --no-verbose: show verbose info on differences on commits (off by default)"
	printf '\t%s\n' "-g, --generate, --no-generate: Generate patches (need CAP_SYS_ADMIN aka sudo privileges as we need to bind mount .git) (off by default)"
	printf '\t%s\n' "--help: Prints help"
	printf '\n%s\n' "Find which commit your no-git friend is working on and generate patches for attaching their works onto git tree"
}


# The parsing of the command-line
parse_commandline()
{
	_positionals_count=0
	while test $# -gt 0
	do
		_key="$1"
		case "$_key" in
			# We support whitespace as a delimiter between option argument and its value.
			# Therefore, we expect the --branch or -b value.
			# so we watch for --branch and -b.
			# Since we know that we got the long or short option,
			# we just reach out for the next argument to get the value.
			-b|--branch)
				test $# -lt 2 && die "Missing value for the optional argument '$_key'." 1
				_arg_branch="$2"
				shift
				;;
			# We support the = as a delimiter between option argument and its value.
			# Therefore, we expect --branch=value, so we watch for --branch=*
			# For whatever we get, we strip '--branch=' using the ${var##--branch=} notation
			# to get the argument value
			--branch=*)
				_arg_branch="${_key##--branch=}"
				;;
			# We support getopts-style short arguments grouping,
			# so as -b accepts value, we allow it to be appended to it, so we watch for -b*
			# and we strip the leading -b from the argument string using the ${var##-b} notation.
			-b*)
				_arg_branch="${_key##-b}"
				;;
			# The verbose argurment doesn't accept a value,
			# we expect the --verbose or -v, so we watch for them.
			-v|--no-verbose|--verbose)
				_arg_verbose="on"
				test "${1:0:5}" = "--no-" && _arg_verbose="off"
				;;
			# We support getopts-style short arguments clustering,
			# so as -v doesn't accept value, other short options may be appended to it, so we watch for -v*.
			# After stripping the leading -v from the argument, we have to make sure
			# that the first character that follows coresponds to a short option.
			-v*)
				_arg_verbose="on"
				_next="${_key##-v}"
				if test -n "$_next" -a "$_next" != "$_key"
				then
					{ begins_with_short_option "$_next" && shift && set -- "-v" "-${_next}" "$@"; } || die "The short option '$_key' can't be decomposed to ${_key:0:2} and -${_key:2}, because ${_key:0:2} doesn't accept value and '-${_key:2:1}' doesn't correspond to a short option."
				fi
				;;
			# See the comment of option '--verbose' to see what's going on here - principle is the same.
			-g|--no-generate|--generate)
				_arg_generate="on"
				test "${1:0:5}" = "--no-" && _arg_generate="off"
				;;
			# See the comment of option '-v' to see what's going on here - principle is the same.
			-g*)
				_arg_generate="on"
				_next="${_key##-g}"
				if test -n "$_next" -a "$_next" != "$_key"
				then
					{ begins_with_short_option "$_next" && shift && set -- "-g" "-${_next}" "$@"; } || die "The short option '$_key' can't be decomposed to ${_key:0:2} and -${_key:2}, because ${_key:0:2} doesn't accept value and '-${_key:2:1}' doesn't correspond to a short option."
				fi
				;;
			# See the comment of option '--verbose' to see what's going on here - principle is the same.
			--help)
				print_help
				exit 0
				;;
			*)
				_last_positional="$1"
				_positionals+=("$_last_positional")
				_positionals_count=$((_positionals_count + 1))
				;;
		esac
		shift
	done
}


# Check that we receive expected amount positional arguments.
# Return 0 if everything is OK, 1 if we have too little arguments
# and 2 if we have too much arguments
handle_passed_args_count()
{
	local _required_args_string="'original' and 'leaf'"
	test "${_positionals_count}" -ge 2 || _PRINT_HELP=yes die "FATAL ERROR: Not enough positional arguments - we require exactly 2 (namely: $_required_args_string), but got only ${_positionals_count}." 1
	test "${_positionals_count}" -le 2 || _PRINT_HELP=yes die "FATAL ERROR: There were spurious positional arguments --- we expect exactly 2 (namely: $_required_args_string), but got ${_positionals_count} (the last one was: '${_last_positional}')." 1
}


# Take arguments that we have received, and save them in variables of given names.
# The 'eval' command is needed as the name of target variable is saved into another variable.
assign_positional_args()
{
	local _positional_name _shift_for=$1
	# We have an array of variables to which we want to save positional args values.
	# This array is able to hold array elements as targets.
	# As variables don't contain spaces, they may be held in space-separated string.
	_positional_names="_arg_original _arg_leaf "

	shift "$_shift_for"
	for _positional_name in ${_positional_names}
	do
		test $# -gt 0 || break
		eval "$_positional_name=\${1}" || die "Error during argument parsing, possibly an Argbash bug." 1
		shift
	done
}

# Now call all the functions defined above that are needed to get the job done
parse_commandline "$@"
handle_passed_args_count
assign_positional_args 1 "${_positionals[@]}"


if [[ "${_arg_generate}" == "on" ]]; then
hash sudo 2>/dev/null || { echo >&2 "Required command 'sudo' is not installed. Aborting."; exit 1; }
hash mount 2>/dev/null || { echo >&2 "Required command 'mount' is not installed. Aborting."; exit 1; }
hash umount 2>/dev/null || { echo >&2 "Required command 'umount' is not installed. Aborting."; exit 1; }
fi

BRANCH="${_arg_branch}"

ORIGINAL_GIT=$(realpath "${_arg_original}")
GIT_LEAF=$(realpath "${_arg_leaf}")
ACT_PATH=$(pwd)

history_hashes=( $(cd "$ORIGINAL_GIT" && git checkout ${BRANCH} 2> /dev/null > /dev/null && git log --pretty=format:"%h") )

if [[ -d "${GIT_LEAF}/.git" ]]; then
    echo "It seems that a .git exists in the directory you want to find. We'll move it to /tmp/$$ for you now and then move it back after checking it."
    mv "${GIT_LEAF}/.git" "/tmp/$$"
fi

if [ -e "${GIT_LEAF}/.git" ]; then
    echo "It seems that a .git file exists in the directory you want to find. Consider replace it to make git work."
    exit 1
fi


min="${history_hashes[0]}"
min_line="-1"

for hash in ${history_hashes[@]}; do
    pushd "$ORIGINAL_GIT" > /dev/null
        git checkout "$hash" 2> /dev/null

        linediff=( $(diff --unified --recursive --new-file \
            --no-dereference \
            --ignore-all-space --ignore-blank-lines --ignore-file-name-case \
             --exclude ".git" \
             --minimal \
             "." "$GIT_LEAF" 2> /dev/null | diffstat | tail -n 1 | awk -v RS='[0-9]+' '$0=RT' ) )

        if [[ "${min_line}" -ge "${linediff[0]}" ]] || [[ "${min_line}" -eq "-1" ]]; then
            min_line="${linediff[0]}"
            min="$hash"
        fi
        if [[ "${_arg_verbose}" == "on" ]]; then
            echo "$hash: ${linediff[0]}, +${linediff[1]}, -${linediff[2]}"
        fi
    popd > /dev/null
done

echo "Minimum commit SHA-1: ${min} with minimum changes: ${min_line} loc."

if [[ "${_arg_generate}" == "on" ]]; then
    mkdir "${GIT_LEAF}/.git"
    sudo mount --bind "${ORIGINAL_GIT}/.git" "${GIT_LEAF}/.git"

    pushd "$ORIGINAL_GIT" > /dev/null
        git checkout "$min" 2> /dev/null
        pushd "${GIT_LEAF}" > /dev/null
            git add . > /dev/null
            if output=$(git status --porcelain 2> /dev/null) && [[ -z "$output" ]]; then
                echo "There are no changes."
            else
                git commit -m "Patch from ${min}" > /dev/null
                commit_SHA=$(git rev-parse HEAD)
                git format-patch -1 "${commit_SHA}" -o "${ACT_PATH}"
            fi
        popd > /dev/null
        [[ ! -z "$output" ]] && git reset --hard HEAD^ > /dev/null
        git checkout ${BRANCH} 2> /dev/null > /dev/null
    popd > /dev/null

    sudo umount "${GIT_LEAF}/.git"
    rm -r "${GIT_LEAF}/.git"
fi

if [ -d "/tmp/$$" ]; then
    echo "Recovering .git directory..."
    mv "/tmp/$$" "${GIT_LEAF}/.git"
fi
