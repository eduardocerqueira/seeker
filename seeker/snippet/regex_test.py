#date: 2022-04-11T16:57:07Z
#url: https://api.github.com/gists/33046de327e90d3e6366ea5b731fa903
#owner: https://api.github.com/users/Nanrech

import re

s = r"""

Traceback (most recent call last):
```
    File "c:\Users\Nan\Documents\bot.py", line 24, in reload

        bot.reload(cog)

    [Previous line repeated 984 more times]

```
"""
print(re.sub(r'```([^```]*)```', string=s, repl=""))
