#date: 2022-03-15T16:51:28Z
#url: https://api.github.com/gists/917875679d5979405a1851d165104318
#owner: https://api.github.com/users/ccwu0918

# I add a few helpers
divs = wd.select("div") # css selecter
div = divs[0]
span = div.select1("span") # return the first result
wd   # screenshot
