#date: 2022-04-14T16:58:54Z
#url: https://api.github.com/gists/778a07c46ce65ebdd26c18579e1925ca
#owner: https://api.github.com/users/OneOger

n =  int(input())
reps = list(map(int,input().split()))
chest = []
biceps = []
back = []
for i in range(n):
    if i % 3 == 0:
        chest.append(reps[i])
    elif i % 3 == 1:
        biceps.append(reps[i])
    else:
        back.append(reps[i])
if sum(chest) > sum(biceps) and sum(chest) > sum(back):
    print('chest')
elif sum(biceps) > sum(chest) and sum(biceps) > sum(back):
    print('biceps')
else:
    print('back')
    