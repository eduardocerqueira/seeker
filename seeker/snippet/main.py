#date: 2023-01-04T16:41:03Z
#url: https://api.github.com/gists/abc7c80d066f08e98e0b62727609f4d7
#owner: https://api.github.com/users/iamdebangshu

from func_2 import verify, compare

d = input("Please enter the first number:")
s = input("Plese enter the second number:")

if verify(d) and verify(s):
    print(compare(d,s))
else:
    print("Invalid input")
