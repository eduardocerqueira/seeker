#date: 2022-06-20T17:01:05Z
#url: https://api.github.com/gists/5c3b912384a2b36044759196a997a9d7
#owner: https://api.github.com/users/HousniBouchen

a = int(input("Type a number: "))
b = int(input("Type a number: "))

c=1
while c!=0:
    c=a%b
    a=b
    b=c

print("The greatest common divisor is: ",a)