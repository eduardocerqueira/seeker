#date: 2022-04-01T17:06:42Z
#url: https://api.github.com/gists/9e2c8992590fbd4486a0776e17a6209e
#owner: https://api.github.com/users/jvoss

#!/usr/bin/env python

# ExaBGP route generator
# Jonathan P. Voss
# 4/1/2022
#

from ipaddress import ip_network

import re
import sys
import time

nexthop    = sys.argv[1]
max_routes = sys.argv[2]

def announce(prefix, nexthop):
  sys.stdout.write('announce route %s next-hop %s\n' % (prefix.with_prefixlen, nexthop))
  sys.stdout.flush()
  time.sleep(0.005)

def build_generators():
  blocks = []

  for x in range(1, 255):
    blocks.append(ip_network('%d.0.0.0/8' % x).subnets(new_prefix=24))

  return blocks

def main():
  block = list(build_generators())
  current_block = list(block.pop(0))

  for x in range(0, int(max_routes)):
    try:
      announce(current_block.pop(0), nexthop)
    except IndexError:
      current_block = list(block.pop(0))
      continue

main()