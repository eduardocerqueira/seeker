#date: 2021-09-16T17:12:08Z
#url: https://api.github.com/gists/0074f322b3af8df76a2bf8e444999187
#owner: https://api.github.com/users/Marcin648

#
# Simple script that prints all process arguments
# usage: python3 check_args.py <PID>
# by Marcin648
# September 2021
#

import sys

if(len(sys.argv) < 2):
    print("usage: check_args.py <PID>")
    exit(1)

pid = sys.argv[1]
cmdline = open("/proc/%s/cmdline" % pid, "r").read()

args = cmdline.split("\0")
for i, arg in enumerate(args):
    print("argv[%d] = %s" % (i, arg))
