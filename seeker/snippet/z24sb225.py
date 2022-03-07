#date: 2022-03-07T17:11:45Z
#url: https://api.github.com/gists/d8ac847c63dc4b21b00b1126a3f5598e
#owner: https://api.github.com/users/alexei-math

f = open('24-v5-9.txt')
s = f.readline()

k = 1
max_k = 1

for i in range(1, len(s)):
    if s[i] >= s[i-1]:
        k += 1
        max_k = max(k, max_k)
    else:
        k = 1

print(max_k)