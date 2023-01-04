#date: 2023-01-04T16:41:03Z
#url: https://api.github.com/gists/abc7c80d066f08e98e0b62727609f4d7
#owner: https://api.github.com/users/iamdebangshu

def verify(a):
    try:
        b = int(a)
        return True
    except:
        return False

def compare(a,b):
    a = int(a)
    b = int(b)
    if (a==b):
        return f'Both are same'
    elif (a>b):
        return f'{a} is larger'
    else:
        return f'{b} is larger'

