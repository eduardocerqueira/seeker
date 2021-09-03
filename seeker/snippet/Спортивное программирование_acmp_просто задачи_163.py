#date: 2021-09-03T17:14:11Z
#url: https://api.github.com/gists/4594a46860bce1018353c5f615ebefc3
#owner: https://api.github.com/users/Maine558

string = input()
x = 0
if string[1] == "+":
    if string[0] == "x":
        x = int(string[4]) - int(string[2])
    else:
        x = int(string[4]) - int(string[0])
else:
    if string[0] == "x":
        x = int(string[4]) + int(string[2])
    else:
        x = -(int(string[4]) - int(string[0]))


print(x)