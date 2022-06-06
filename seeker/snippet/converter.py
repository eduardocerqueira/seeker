#date: 2022-06-06T16:48:59Z
#url: https://api.github.com/gists/a8597ea63960864d4728ae9bc36eb623
#owner: https://api.github.com/users/raresteak

#!/usr/bin/env python3
# Author raresteak
# basic leetspeak converter
# Converts plain text
import sys
userInput = sys.argv[1:]
if len(userInput) == 0:
    print("USAGE: " + sys.argv[0] + " plain text")
    sys.exit()
text=str(' '.join(userInput))
text=text.replace('a', '@')
text=text.replace('b', '6')
text=text.replace('g', '9')
text=text.replace('l', '1')
text=text.replace('s', '5')
text=text.replace('t', '+')
text=text.replace('A', '4')
text=text.replace('B', '8')
text=text.replace('E', '3')
text=text.replace('I', '|')
text=text.replace('J', ']')
text=text.replace('O', '0')
text=text.replace('S', '$')
text=text.replace('T', '7')
text=text.replace('Z', '2')
print(text)