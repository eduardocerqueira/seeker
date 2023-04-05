#date: 2023-04-05T16:52:39Z
#url: https://api.github.com/gists/0f92be767bf7f2fb1448d8fe58470847
#owner: https://api.github.com/users/akshara-a

n = input()

i = 0

for j in range(len(n)-1,-1,-1):
    res = n[i]*int(n[j])
    if int(n[j]) == 0:
        print('-')
    else:
        print(res)
    i += 1
    
    
# Sample I/O
# n = 56743
# 555
# 6666
# 7777777
# 444444
# 33333


# n = 4509
# 444444444
# -
# 00000
# 9999   