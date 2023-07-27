#date: 2023-07-27T17:00:13Z
#url: https://api.github.com/gists/22b0a6f10fbc220b5ee30e07a59398b1
#owner: https://api.github.com/users/cinder-star

s, k = map(str, input().split())
k = int(k)

ln = len(s)

if ln < k:
    s = s[::-1]
elif ln < 2 * k:
    s = s[:k][::-1] + s[k:]
else:
    reversed_string = ""
    counter = 0
    steps = ln // (2 * k) + (0 if ln % (2 * k) == 0 else 1)
    for i in range(steps):
        reverse_range = min(counter + k, ln)
        max_range = min(counter + 2 * k, ln)
        reversed_string = (
            reversed_string
            + s[counter:reverse_range][::-1]
            + s[reverse_range:max_range]
        )
        counter += 2 * k
    s = reversed_string

print(s)
