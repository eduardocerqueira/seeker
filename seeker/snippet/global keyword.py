#date: 2021-11-12T17:01:02Z
#url: https://api.github.com/gists/71165865fafd3975b98fcf1da48934af
#owner: https://api.github.com/users/KananMahammadli

a = 5
def func():
    global a
    a = 20
    
func()
print(a)