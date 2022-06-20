#date: 2022-06-20T16:53:24Z
#url: https://api.github.com/gists/30a77b3fdb390fa8df3f288d8c1b54f1
#owner: https://api.github.com/users/HousniBouchen

c = input("Type a character: ")
if c.isalpha():
    print(c, " is letter !\n")
elif c.isdigit():
    print(c, " is digit !\n")
else:
    print(c, " is a special character !\n")