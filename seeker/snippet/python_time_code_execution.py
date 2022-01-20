#date: 2022-01-20T17:11:40Z
#url: https://api.github.com/gists/67178f8a3f8760899a3473cf536ea46d
#owner: https://api.github.com/users/cibelesimoes

##Begin
import time
from datetime import timedelta
start = time.time()
#

### The code that you want to measure


#
end = time.time()
elapsed = end - start
print(timedelta(seconds=elapsed)) ## will print as HH:mm:ss:mmmmmmmm
##End