#date: 2022-06-20T16:57:06Z
#url: https://api.github.com/gists/f8d388f34cdbfc6a6f93fe1ba4e906fc
#owner: https://api.github.com/users/HousniBouchen


a = input("Type a string: ")

n=0
i=0
while i<len(a):
    if a[i].isdigit():
        n = n + 1
    i = i + 1

print("Number of digit: ", n)