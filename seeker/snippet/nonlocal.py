#date: 2021-11-12T17:08:37Z
#url: https://api.github.com/gists/f1e5a0ce9184dadf847b4d9b08dbf89b
#owner: https://api.github.com/users/KananMahammadli

# accessing to global variable from inner local scope
a = 10
def outer():
    def inner():
        print(a)
    inner()

print(outer())