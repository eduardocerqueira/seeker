#date: 2022-09-19T17:16:30Z
#url: https://api.github.com/gists/91cd04ff06e074fb63d558c31c85b0cf
#owner: https://api.github.com/users/s-hiiragi

def print_attrs(value):
    from functools import reduce
    
    mro = type(value).mro()
    attrsets = [set(dir(x)) for x in mro] + [set()]
    ownattrsets = [x - attrsets[i+1] for i,x in enumerate(attrsets) if i < len(mro)]
    
    for type_, ownattrset in zip(mro, ownattrsets):
        print(type_.__name__)
        print(' ', list(ownattrset))
