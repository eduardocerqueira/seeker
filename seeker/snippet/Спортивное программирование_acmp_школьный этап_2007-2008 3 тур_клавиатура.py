#date: 2021-09-03T17:14:11Z
#url: https://api.github.com/gists/4594a46860bce1018353c5f615ebefc3
#owner: https://api.github.com/users/Maine558

alf = "qwertyuiopasdfghjklzxcvbnm"

N = input()

for i in range(len(alf)):
    if N == alf[i] and N != "m":
        print(alf[i+1])
if N == "m":
    print("q")