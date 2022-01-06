#date: 2022-01-06T17:01:44Z
#url: https://api.github.com/gists/81d5a0b99e65bfb5ce1e2e06e1eee9d3
#owner: https://api.github.com/users/narenaryan

import asyncio

# A co-routine
async def add(x: int, y: int):
  return x + y

# An event loop
loop = asyncio.get_event_loop()

# Pass the co-routine to the loop
result = loop.run_until_complete(add(3, 4))
print(result) # Prints 7
