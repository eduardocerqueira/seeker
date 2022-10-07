#date: 2022-10-07T17:14:21Z
#url: https://api.github.com/gists/3880409783c9f666cdc4341a4c3a66cd
#owner: https://api.github.com/users/felixbd

#!/usr/bin/env python3

import math

def base_10_to_base_2(int: a) -> int:
	rv = ""
	while a / 2 != 0:
		rv += str(math.ceil((a / 2) % 1))
		a = a // 2
	return str(rv[::-1])

# or
# https://itnik.net/blog/How%20to%20Implement%20the%20Square%20and%20Multiply%20Algorithm%20in%20Python
# 07/10/2022
# base_10_to_base_2_v2 = lambda e: [(e >> bit) & 1 for bit in range(e.bit_length() - 1, -1, -1)]