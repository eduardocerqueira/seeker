#date: 2024-11-11T17:02:00Z
#url: https://api.github.com/gists/88e0e49bb8ad1c0486ae22569c2fdbd7
#owner: https://api.github.com/users/kielmarj

#!/usr/bin/env bash

# ------------------------------------------------------------------------------
# FILE: bash-bible.sh
# DATE: 2024-11-11_07:46 | MODIFIED: 2024-11-11_11:43
# ABOUT: Functions based on pure-bash-bible. Modifications include shortened
#   function names & formatting adjustments for readability.
#   See https://github.com/dylanaraps/pure-bash-bible for the original work.
#   *WIP*, many more to add still.
# AUTHOR: pure-bash-bible © 2018 Dylan Araps
#   Modifications © 2024 Jessica Kielmar (GitHub: kielmarj)
# LICENSE: MIT License (See LICENSE file for details)
# ------------------------------------------------------------------------------

help() {
	cat <<EOF

--help Print this help message. See script for examples.
--trim Trim all white-space from string & truncate spaces
  Usage: trim "   example   string    "
--regex Use regex on a string 
  Usage: regex "string" "regex"
--hex Validate a hex color
  Usage: hex "color"
--split Split a string on a delimiter
  Usage: split "string" "delimiter"
--lower Change a string to lowercase
  Usage: lower "string"
--upper Change a string to uppercase
  Usage: upper "string"
--strip Strip all instances of pattern from string
  Usage: strip "string" "pattern"
--contains Check if string contains a sub-string
  Usage: contains "string" "pattern"

EOF
}

##                                                    ##
## Trim all white-space from string & truncate spaces ##
##                                                    ##
trim() {
    # Usage: trim "   example   string    "
    set -f
    set -- $*
    printf '%s\n' "$*"
    set +f
}

##                       ##
## Use regex on a string ##
##                       ##
regex() {
    # Usage: regex "string" "regex"
    [[ "$1" =~ $2 ]] && printf '%s\n' "${BASH_REMATCH[1]}"
}

##                      ##
## Validate a hex color ##
##                      ##
hex() {
    # Usage: hex "#FFFFFF"
    if [[ $1 =~ ^(#?([a-fA-F0-9]{6}|[a-fA-F0-9]{3}))$ ]]; then
        printf '%s\n' "$1 is a valid hex color."
    else
        printf '%s\n' "Error: $1 is not a valid hex color."
        return 1
    fi
}

##                               ##
## Split a string on a delimiter ##
##                               ##
split() {
    # Usage: split "string" "delimiter"
    IFS=$'\n' read -d "" -ra arr <<< "${1//$2/$'\n'}"
    printf '%s\n' "${arr[@]}"
}

##                              ##
## Change a string to lowercase ##
##                              ##
lower() {
    # Usage: lower "string"
    printf '%s\n' "${1,,}"
}

##                              ##
## Change a string to uppercase ##
##                              ##
upper() {
    # Usage: upper "string"
    printf '%s\n' "${1^^}"
}

##                                            ##
## Strip all instances of pattern from string ##
##                                            ##
strip() {
    # Usage: strip "string" "pattern"
    printf '%s\n' "${1//$2}"
}

##                                       ##
## Check if string contains a sub-string ##
##                                       ##
contains() {
    # Usage: contains "string" "pattern"
    if [[ $1 == *$2* ]]; then
        printf '%s\n' "Contains \"$2\""
    else
        printf '%s\n' "Does not contain \"$2\""
    fi
}

# Main case statementto parse the command-line options
case "$1" in
    --help)
        help
    ;;
    --trim)
        trim "$2"
    ;;
    --regex)
        regex "$2" "$3"
    ;;
    --hex)
        hex "$2"
    ;;
    --split)
        split "$2" "$3"
    ;;
    --lower)
        lower "$2"
    ;;
    --upper)
        upper "$2"
    ;;
    --strip)
        strip "$2" "$3"
    ;;
    --contains)
        contains "$2" "$3"
    ;;
    *)
        printf 'Invalid option: %s\n' "$1"
        printf 'Use --help to see available options.\n'
        exit 1
    ;;
esac
