#date: 2021-11-30T17:10:47Z
#url: https://api.github.com/gists/69f270ca68522a5e422fcf3533c9b8b3
#owner: https://api.github.com/users/CodeShane

#!/usr/bin/env bash
# Copyright (c) 2021 Shane Robinson
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

# use `logt` to "tail -n30" the current log file
JOURNALLOGPATH="${JOURNALLOGPATH:-~/work-logs}"
echo "$JOURNALLOGPATH"

mkdir -p "$JOURNALLOGPATH"
cd "$JOURNALLOGPATH" || exit 1

mkdir -p "$JOURNALLOGPATH"
logfile="$JOURNALLOGPATH/$(date +"%Y-%m").txt"

if [[ -z $1 ]]; then
  echo "    \"$0\" Simple timestamped event log journal utility for your path."
  echo;
  echo "    Example Usage - add records:"
  echo "    \$ $0 Message, prefixed with date+time, to append to file: $logfile"
  echo "    \$ $0 did something cool."
  echo "    \$ $0 'can be quoted, of course'"
  echo;
  echo "    Usage - help and tail (this):"
  echo "    \$ $0"
  echo "    -----------------------------"
  echo;
  echo "    \$ tail -n30 $logfile"
  tail -n30 "$logfile"
  
  exit 0
fi

echo "$(date +'%Y-%m-%d.%H%M.%S')   $@" >> $logfile

# This is echoed back so you can notice any parsing errors.
echo "-- Logged: " $@
echo "-- Logfile: $logfile"
