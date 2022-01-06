#date: 2022-01-06T17:19:38Z
#url: https://api.github.com/gists/920ec78d22c3a7174aac063034217708
#owner: https://api.github.com/users/narenaryan

import asyncio

# A co-routine
async def add(x: int, y: int):
  return x + y

# An event loop
loop = asyncio.get_event_loop()

# Execute two co-routines
result1 = loop.run_until_complete(add(3, 4))
result2 = loop.run_until_complete(add(5, 5))

print(result1) # Prints 7
print(result2) # Prints 10