#date: 2022-08-25T17:05:30Z
#url: https://api.github.com/gists/ad5e97d7c70a79fad702632030e46a07
#owner: https://api.github.com/users/paranlee

# !/usr/bin/python
# -*- encoding: utf-8 -*-

import sys

dp = []
for i in range(100):
    dp.append(0)
dp[0] = 0
dp[1] = 1
dp[2] = 2
for i in range(3, 100):
    dp[i] = dp[i - 1] + dp[i - 2]
print dp[int(sys.argv[1])]