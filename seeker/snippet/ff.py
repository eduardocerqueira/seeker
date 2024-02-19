#date: 2024-02-19T17:09:35Z
#url: https://api.github.com/gists/65e0be2dffe3c5e0d76069900e465000
#owner: https://api.github.com/users/powellnathanj

#!/usr/bin/env python3
import sys

ffilter = ""

for line in sys.stdin:
    nl = line.rstrip()
    ffilter = ffilter + "( addr.dst in '" + nl + "' ) or "

print(ffilter)