#date: 2021-12-31T17:16:33Z
#url: https://api.github.com/gists/5c626fcaa7e6e8e6a1587baf313f3485
#owner: https://api.github.com/users/rainyman2012

import numpy as np

day_price = np.array([[0,7], [1,12], [2,5], [3,3], [4,11], [5, 6], [6,10] ,[7,2], [8,0]])

results = np.zeros((9, 9))

print(results)
for buy in day_price:
    for sell in day_price:
        if buy[0] < sell[0]:
            results[buy[0],sell[0]] = sell[1] - buy[1]
        
print(results)
all_results = np.where(results == np.max(results))

for res in list(zip(all_results[0], all_results[1])):
    print("you should buy in", res[0], "and sell in", res[1], "your profit is", results[res])
