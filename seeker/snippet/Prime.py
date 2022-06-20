#date: 2022-06-20T17:02:44Z
#url: https://api.github.com/gists/3835d89fb4deb2f5c904cb4e5474a66c
#owner: https://api.github.com/users/HousniBouchen

n=int(input("Type a positive number: "))
while n<0:
    n=int(input("Type a positive number: "))

i=2
while i<n:
    if n%i==0:
        break
    i = i + 1

if i==n or n==0 or n==1:
    print(n, " is prime \n")
else:
    print(n, " is not prime \n")