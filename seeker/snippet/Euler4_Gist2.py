#date: 2022-02-21T16:57:11Z
#url: https://api.github.com/gists/78470b436176f7de305e2333a49bd078
#owner: https://api.github.com/users/glantonb

def is_palindrome(x):
    x = str(x)

    return x == ''.join(reversed(x))