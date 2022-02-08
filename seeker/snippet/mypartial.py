#date: 2022-02-08T17:10:48Z
#url: https://api.github.com/gists/14de8731807a44e4cb5e82233952fa45
#owner: https://api.github.com/users/KommuSoft

def partial(f, *args, **kwargs):
    def g(*args2, **kwargs2):
        return f(*args, *args2, **kwargs, **kwargs2)
    return g