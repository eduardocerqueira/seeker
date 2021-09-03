#date: 2021-09-03T17:14:11Z
#url: https://api.github.com/gists/4594a46860bce1018353c5f615ebefc3
#owner: https://api.github.com/users/Maine558

k = int(input())
a = []
for j in range(k):
    n = input()

    new_10 = 0

    for i in range(len(n)):
        if n[i] == "1":
            new_10 += 2**(len(n)-i-1)

    a.append(new_10)

for i in range(k):
    if a[i]%7 == 0:
        print("Yes")
    else:
        print("No")