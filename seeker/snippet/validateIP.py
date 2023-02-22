#date: 2023-02-22T17:04:39Z
#url: https://api.github.com/gists/373db354fe882aa215ff82191de00bd1
#owner: https://api.github.com/users/Varsha-R

import re

regex = "^(([0-9]|[1-9][0-9]|1[0-9][0-9]|2[0-4][0-9]|25[0-5])\.){3}([0-9]|[1-9][0-9]|1[0-9][0-9]|2[0-4][0-9]|25[0-5])$"

def validateIP(ipAddress):
    if re.search(regex, ipAddress):
        return True
    return False

print(validateIP("192.168.0.13"))
print(validateIP("110.234.52.0"))
print(validateIP("366.1.2.2"))

# Reference - https://www.regular-expressions.info/numericranges.html#:~:text=To%20match%20all%20characters%20from,That's%20the%20easy%20part.
# regex = "^((25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9][0-9]|[0-9])\.){3}(25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9][0-9]|[0-9])$"