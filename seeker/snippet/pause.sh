#date: 2024-06-21T17:11:40Z
#url: https://api.github.com/gists/cb5935d0d507ad54ad1439617754ea2b
#owner: https://api.github.com/users/imdebating

# Pause processing for n seconds. Used for debugging.
# Prettily print the time by adding a dot (.) every
# .25 seconds. Also hides the curson from view for
# clarity.
function pause(){
  # $1: A string to be printed at the beginning of the output
  # $2: The number of seconds to pause
  printf "\e[?25l" # Hide the cursor
  for ((i=$2; i >= 1; i--)) ; do 
    printf "%s Holding for %i      \r" "$1" "$i"; sleep .25
    printf "%s Holding for %i.     \r" "$1" "$i"; sleep .25
    printf "%s Holding for %i..    \r" "$1" "$i"; sleep .25
    printf "%s Holding for %i...   \r" "$1" "$i"; sleep .25
  done
  printf "%s Hold completed...resuming operations.     \n" "$1"
  printf "\e[?25h" # Show the cursor
}
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#                                                           #
#  888888 8888888              88 8888888                   #
#    88   88                   88 88    oo                  #
#    88   88                   88 88                        #
#    88   88888 .d8b.   .d8b.  88 88888 88 8888b.  .d8b.    #
#    88   88   d8P Y8b d8P Y8b 88 88    88 88  8b d8P Y8b   #
#    88   88   8888888 8888888 88 88    88 88  88 8888888   #
#    88   88   Y8b.    Y8b.    88 88    88 88  88 Y8b.      #
#  888888 88    ºY888P  ºY888P 88 88    88 88  88  ºY888P   #
#                           (c) 2015-2024 I Feel Fine, Inc. #
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# pause.sh
# -------------
# Description: 
# Bash function that pauses for n seconds.
# -------------
# MIT License
# 
# Copyright (c) 2024 I Feel Fine, Inc.
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