#date: 2024-04-25T17:00:14Z
#url: https://api.github.com/gists/39f3419e5e7d74410b53bdb5ee505fb2
#owner: https://api.github.com/users/Garfounkel

#!/bin/bash
#================================================================================
# MIT License
#
# Copyright (c) 2024 Simon Andersen
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#================================================================================

_parse_params() {
    : 'Usage: parse_params "--arg1,--arg2,--arg3" "$@"'
    : 'Parse command line arguments into params and rest associative arrays'
    : 'params contains arguments that are in the list, ex: arg1, arg2, arg3'
    : 'rest contains every other arguments that are not in the list'
    : 'They can then be accessed using ${params[arg1]}, ${rest[@]}'

    # Reset params and rest to avoid conflicts with previous calls
    declare -gA params=()
    declare -g rest=()

    # Extract the list of known arguments
    local list=",${1},"
    shift

    local arg
    local value
    local next_arg
    
    # Loop through all arguments
    while [[ $# -gt 0 ]]; do
        arg="${1}"
        next_arg="${2}"

        # Check if the next argument is a value or a new argument
        if [[ "$next_arg" != --* ]]; then
            value="$next_arg"
            shift # additional shift for the value
        else
            value=""
        fi

        # Check if the argument is in the known list
        if [[ "$list" == *",${arg},"* ]]; then
            # Argument is known, store its value in params
            params["${arg:2}"]="$value"
        else
            # Argument is not known, store it in rest
            if [[ -n "$value" ]]; then
                rest+=("$arg" "$value")
            else
                rest+=("$arg")
            fi
        fi
        shift # Move to the next argument
    done
}


### Exemple usage ###
_exemple_usage() 
(
    _print_all() {
        for arg in "$@"; do
            echo print_all: "$arg"
        done
    }

    _test() {
        parse_params "--foo,--bar" "$@"

        # --foo and --bar from the list are stored in params
        echo "params[foo]: ${params[foo]}"
        echo "params[bar]: ${params[bar]}"
        echo
        # --baz is not in the list, it is stored in rest
        echo "params: ${params[@]}"
        echo "rest: ${rest[@]}"
        echo
        # pass rest to a function like if it was $@
        _print_all "${rest[@]}"
    }
    _test --foo foo_value --baz baz_value --bar bar_value --foobar
)

# Uncomment this to test the exemple:
# _exemple_usage
