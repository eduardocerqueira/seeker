#date: 2021-11-12T17:13:50Z
#url: https://api.github.com/gists/776fdc5036b244bdab73d8541768f3e6
#owner: https://api.github.com/users/KananMahammadli

def outer():
    a  = 10
    def inner():
        print(a)
    inner()

outer()