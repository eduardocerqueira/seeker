#date: 2022-01-05T16:56:53Z
#url: https://api.github.com/gists/95ada323a636b378b573801cb078fcf9
#owner: https://api.github.com/users/cravatsc

a = [1,2,3,4,5]
b = [4,5,6,7,8]
c = [number for number in a if number in b]
print(c)
# [4, 5] <- expected outcome