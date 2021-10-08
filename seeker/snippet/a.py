#date: 2021-10-08T17:14:38Z
#url: https://api.github.com/gists/04cebde3d8028586cea8823438db99c5
#owner: https://api.github.com/users/MrPanch

n = int(input())
k = int(input())
direction = input()
a = []
for i in range(n):
    a.append(int(input()))
    
result = [0] * n
k %= n

if direction == "LEFT":
    k = n - k
    
result[k] = a[0]
i = 1
j = k+1
while j != k:
    result[j] = a[i]
    i += 1
    j = (j + 1) % n

print(result)