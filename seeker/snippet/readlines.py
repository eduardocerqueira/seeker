#date: 2022-11-04T16:57:45Z
#url: https://api.github.com/gists/32fa4c3f520fc9c17d36e272487a2aa8
#owner: https://api.github.com/users/edanursunay

with open("surname.txt", "r") as f:
    lines = f.readlines()
    for i in lines:
        print(i)