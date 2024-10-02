#date: 2024-10-02T17:09:36Z
#url: https://api.github.com/gists/ffd4c3498a9a85817358a8f799fe9f05
#owner: https://api.github.com/users/infoxmaz

def test_function ():
    def inner_function ():
        print('Я в области видимости функции test_function')
    inner_function()
test_function()
inner_function()