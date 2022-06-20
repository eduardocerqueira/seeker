#date: 2022-06-20T16:58:23Z
#url: https://api.github.com/gists/20ffddd950834a96d26bf5fa14ddfc2e
#owner: https://api.github.com/users/HousniBouchen

n = int(input("Type a positive number: "))
while n<0:
    n = int(input("Type a positive number: "))
f=1
i=1
while i<=n:
    f=f*i
    i=i+1

print(n,"!=",f)