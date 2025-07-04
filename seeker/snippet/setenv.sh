#date: 2025-07-04T16:31:43Z
#url: https://api.github.com/gists/96c2446a64e927db34fc94e9716a330f
#owner: https://api.github.com/users/vic-dellarocco

#!/bin/bash
# Easy script to create a python venv and install requirements
# or run the venv if it already exists.


# Copyright (c) 2025 Vic Dellarocco
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without
# restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons
# to whom the Software is furnished to do so, subject to the
# following conditions:
#
# The above copyright notice and this permission notice shall
# be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY
# KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE AUTHORS
# OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
# OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


# VV: I use this as the base for my python programs.
#  Features:
#   * It auto-detects a python venv (any venv, so be careful).
#   * It prompts the user before installing software.
#   * It automatically activates the venv if one exists. This is
#     so nice for the user because then they can run the program
#     without having to source the activate script.
#
# There are two ways to use this:
#  * You can set this up to install/activate a venv by sourcing
#    this file. This is for the programmer.
#  * You can "chmod +x" this file so that the venv is activated
#    in a subshell which then can run a python program. When the
#    subshell exits, the venv will not be active. This is what
#    your users expect.

FULL_PATH_TO_SCRIPT="$(realpath "$0")"
SCRIPT_DIRECTORY="$(dirname "$FULL_PATH_TO_SCRIPT")"

# VENV detection and activation.
 function detect-venv   () {
  python -c 'if 1:#for indentation
  import sys
  ss=sys.prefix != sys.base_prefix
  # print("%s\n%s\n%s" % (sys.prefix,sys.base_prefix,ss) )
  if ss:
   sys.exit(0)
  sys.exit(1)
  '
  return $?
  }
 function activate-venv () {
  # shellcheck disable=SC1091
  source "$SCRIPT_DIRECTORY"/venv/bin/activate || true;
  detect-venv
  VENV=$?
  if [ $VENV = 1 ]; then
   echo "No venv available."
   echo "Need a venv and these python dependencies:"
   echo ""
   sed 's/^/  /' requirements.txt
   echo ""
   read -rp "Do you want to create a python venv and install python dependencies (y/n) " response
   case ${response:0:1} in
    y|Y )
     echo "Creating venv..."
     python -m venv "$SCRIPT_DIRECTORY"/venv
     echo "Activating venv."
     # shellcheck disable=SC1091
     source "$SCRIPT_DIRECTORY"/venv/bin/activate
     # Sanity check, then Install python dependencies:
     detect-venv
     VENV=$?
     if [ $VENV = 1 ]; then
      echo "No active venv. Refusing to procede."
      return 1
      fi
     : \
     && echo 'Installing python libraries.' \
     && python -m pip install -r "$SCRIPT_DIRECTORY"/requirements.txt \
     ;
     ;;
    n|N )
     echo "Ok, no action taken."
     ;;
    * )
     echo "Ok, no action taken."
     ;;
   esac
   unset response
   fi
  unset VENV
  }
 function run-app       () {
  # shellcheck disable=SC2078
  if [ "Run your program:" ]; then
   # Sanity check:
   detect-venv
   VENV=$?
   if [ $VENV = 1 ]; then
    echo "No active venv. Refusing to procede."
    return 1
    fi
   # Put your script logic here:
   echo "Put your script logic here."
   fi
  }

activate-venv
run-app
