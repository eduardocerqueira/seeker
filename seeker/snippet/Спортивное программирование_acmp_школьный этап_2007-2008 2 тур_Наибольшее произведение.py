#date: 2021-09-03T17:14:11Z
#url: https://api.github.com/gists/4594a46860bce1018353c5f615ebefc3
#owner: https://api.github.com/users/Maine558

N = int(input())

a = [int(s) for s in input().split()]


a.sort()

ans = 0

if a[0]*a[1] > a[len(a)-2]*a[len(a)-3]:
    ans = a[0]*a[1]*a[len(a)-1]
else:
    ans = a[len(a)-1]*a[len(a)-2]*a[len(a)-3]
print(ans)