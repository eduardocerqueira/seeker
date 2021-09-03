#date: 2021-09-03T17:14:11Z
#url: https://api.github.com/gists/4594a46860bce1018353c5f615ebefc3
#owner: https://api.github.com/users/Maine558

N = int(input())

height = [int(s) for s in input().split()]

height_of_bus = 437

crash = False

for i in range(N):
    if height[i] <= height_of_bus:
        print("crash",i+1)
        crash = True
        break
    else:
        continue
if crash == False:
    print("No crash")