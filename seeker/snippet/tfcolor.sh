#date: 2022-12-16T16:48:56Z
#url: https://api.github.com/gists/e645502cf4bd4ad380ab77fc0240428b
#owner: https://api.github.com/users/fmaylinch

#!/usr/bin/env bash

# Colors whole line in Terraform changes, so it's more highlighted.
# By deault, uses background color, that is more visible.
# If you specify argument "short", it just shows the changed lines.

# Usage: terraform plan | tfcolor.sh [short]


set -u
# note: with -e grep terminates the script when a line is not found

# TODO: Add option to use foreground/background colors.
#       I could use the option handling I did in Postres scripts.
OPTION=${1:-}

# Bash colors
# https://gist.github.com/JBlond/2fea43a3049b38287e5e9cefc87b2124
reset=$(printf '\33\\[0m')
black=$(printf '\33\\[30m')
red=$(printf '\33\\[31m')
green=$(printf '\33\\[32m')
yellow=$(printf '\33\\[33m')
redbg=$(printf '\33\\[41m')
greenbg=$(printf '\33\\[42m')
yellowbg=$(printf '\33\\[43m')

if [ "$OPTION" = "short" ]; then
  # IFS='' is to keep prefix spaces
  # Maybe there's a simpler way to pipe the input
  while IFS='' read data; do
    # remove color reset, put color reset at the end, filter colored lines
    printf "$data\n" \
      | sed "s/$reset//g" \
      | sed -E "s/(.+)/\1$reset/" \
      | grep -e $green -e $red -e $yellow
  done
else
  while IFS='' read data; do
    # remove color rest, put color reset at the end, replace color with bg color
    printf "$data\n" \
      | sed "s/$reset//g" \
      | sed -E "s/(.+)/\1$reset/" \
      | sed "s/$green/$greenbg$black/" \
      | sed "s/$red/$redbg$black/" \
      | sed "s/$yellow/$yellowbg$black/"
  done
fi
