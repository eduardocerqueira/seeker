#date: 2022-04-05T17:05:21Z
#url: https://api.github.com/gists/21e2f68d042255add3f612dd8a357fab
#owner: https://api.github.com/users/dalar25

print("pick a number between 0 and 100")
minnum = 0
maxnum = 100
win = False
while not win:
    newin = input("Is your number higher or lower than " + str(int(((minnum+maxnum)/2)//1)) + "? ")
    if newin.lower() == "correct":
        win = True
    elif newin.lower() == "higher":
        minnum = ((minnum+maxnum)/2)//1
    else:
        maxnum = ((minnum+maxnum)/2)//1
