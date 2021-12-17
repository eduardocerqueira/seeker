#date: 2021-12-17T16:56:01Z
#url: https://api.github.com/gists/364732600186ef2d20c947539a04e772
#owner: https://api.github.com/users/msyvr

def setpy():
    """
    python sets are collections of objects which are:
    unique
    unordered
    unchangeable*
    * where unchangeable means individual items can't be replaced in place, but items can be removed and new items added
    """

    # unique and unsorted
    a = {4, 4, 2, 3}
    print(a)
    '''
    prints to stdout:
    {2, 3, 4}
    
    nb: duplicate values have been omitted and items presented in ascending order, not the original ordering
    '''
    print(a.sorted())
    '''
    generates error message:
    AttributeError: 'set' object has no attribute 'sorted'
    '''

    # unchangeable
    a[2] = 7
    print(a)
    '''
    generates error message:
    TypeError: 'set' object does not support item assignment
    '''

if __name__ == "__main__":
    # function inputs - optional
    # call function
    setpy()