#date: 2021-09-03T17:14:11Z
#url: https://api.github.com/gists/4594a46860bce1018353c5f615ebefc3
#owner: https://api.github.com/users/Maine558

N = int(input())

coun_1 = 0
coun_0 = 0

for i in range(N):
    coin = int(input())

    if coin == 1:
        coun_1 += 1
    else:
        coun_0 += 1

print(min(coun_1,coun_0))