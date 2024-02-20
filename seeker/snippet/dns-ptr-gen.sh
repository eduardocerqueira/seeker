#date: 2024-02-20T16:49:24Z
#url: https://api.github.com/gists/e693f7b81633f166a52203852481c3f2
#owner: https://api.github.com/users/brhm72

#!/bin/bash

# PTR kaydi olusturma
# 64	IN	PTR	host-192.168.1.64.alanadi.com
# 65	IN	PTR	host-192.168.1.65.alanadi.com
# 66	IN	PTR	host-192.168.1.66.alanadi.com
# 67	IN	PTR	host-192.168.1.67.alanadi.com
# 68	IN	PTR	host-192.168.1.68.alanadi.com
# 69	IN	PTR	host-192.168.1.69.alanadi.com
# 70	IN	PTR	host-192.168.1.70.alanadi.com
# 71	IN	PTR	host-192.168.1.71.alanadi.com

ALIAS="host-192.168.1"
DOMAIN="alanadi.com"

START=64
END=71

for i in $(eval echo "{$START..$END}")
do
  echo -e "$i\tIN\tPTR\t$ALIAS.$i.$DOMAIN"
done

