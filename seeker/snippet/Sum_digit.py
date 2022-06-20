#date: 2022-06-20T17:03:53Z
#url: https://api.github.com/gists/31562467e993261afb1bc1134bc80eda
#owner: https://api.github.com/users/HousniBouchen

a = input("Type a string: ")

n=0
i=0
while i<len(a):
    if a[i].isdigit():
        n = n + int(a[i])
    i = i + 1

print("The sum of digits is: ", n)