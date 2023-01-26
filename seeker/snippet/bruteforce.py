#date: 2023-01-26T16:56:58Z
#url: https://api.github.com/gists/ac3199721c2c82b3454f67259340b7a5
#owner: https://api.github.com/users/Redtooth69

"""
Password brute-force algorithm.

 "**********"L "**********"i "**********"s "**********"t "**********"  "**********"o "**********"f "**********"  "**********"m "**********"o "**********"s "**********"t "**********"  "**********"p "**********"r "**********"o "**********"b "**********"a "**********"b "**********"l "**********"e "**********"  "**********"p "**********"a "**********"s "**********"s "**********"w "**********"o "**********"r "**********"d "**********"s "**********"  "**********"a "**********"n "**********"d "**********"  "**********"e "**********"n "**********"g "**********"l "**********"i "**********"s "**********"h "**********"  "**********"n "**********"a "**********"m "**********"e "**********"s "**********"  "**********"c "**********"a "**********"n "**********"  "**********"b "**********"e "**********"  "**********"f "**********"o "**********"u "**********"n "**********"d "**********", "**********"  "**********"r "**********"e "**********"s "**********"p "**********"e "**********"c "**********"t "**********"i "**********"v "**********"e "**********"l "**********"y "**********", "**********"  "**********"a "**********"t "**********": "**********"
- https: "**********"
- https://github.com/dominictarr/random-name/blob/master/middle-names.txt

Author: Raphael Vallat
Date: May 2018
Python 3
"""
import string
from itertools import product
from time import time
from numpy import loadtxt


 "**********"d "**********"e "**********"f "**********"  "**********"p "**********"r "**********"o "**********"d "**********"u "**********"c "**********"t "**********"_ "**********"l "**********"o "**********"o "**********"p "**********"( "**********"p "**********"a "**********"s "**********"s "**********"w "**********"o "**********"r "**********"d "**********", "**********"  "**********"g "**********"e "**********"n "**********"e "**********"r "**********"a "**********"t "**********"o "**********"r "**********") "**********": "**********"
    for p in generator:
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"' "**********"' "**********". "**********"j "**********"o "**********"i "**********"n "**********"( "**********"p "**********") "**********"  "**********"= "**********"= "**********"  "**********"p "**********"a "**********"s "**********"s "**********"w "**********"o "**********"r "**********"d "**********": "**********"
            print('\nPassword: "**********"
            return ''.join(p)
    return False


 "**********"d "**********"e "**********"f "**********"  "**********"b "**********"r "**********"u "**********"t "**********"e "**********"f "**********"o "**********"r "**********"c "**********"e "**********"( "**********"p "**********"a "**********"s "**********"s "**********"w "**********"o "**********"r "**********"d "**********", "**********"  "**********"m "**********"a "**********"x "**********"_ "**********"n "**********"c "**********"h "**********"a "**********"r "**********"= "**********"8 "**********") "**********": "**********"
    """Password brute-force algorithm.

    Parameters
    ----------
    password : "**********"
        To-be-found password.
    max_nchar : int
        Maximum number of characters of password.

    Return
    ------
    bruteforce_password : "**********"
        Brute-forced password
    """
    print('1) Comparing with most common passwords / first names')
    common_pass = loadtxt('probable-v2-top12000.txt', dtype=str)
    common_names = loadtxt('middle-names.txt', dtype=str)
    cp = "**********"== password]
    cn = "**********"== password]
    cnl = "**********"== password]

    if len(cp) == 1:
        print('\nPassword: "**********"
        return cp
    if len(cn) == 1:
        print('\nPassword: "**********"
        return cn
    if len(cnl) == 1:
        print('\nPassword: "**********"
        return cnl

    print('2) Digits cartesian product')
    for l in range(1, 9):
        generator = product(string.digits, repeat=int(l))
        print("\t..%d digit" % l)
        p = "**********"
        if p is not False:
            return p

    print('3) Digits + ASCII lowercase')
    for l in range(1, max_nchar + 1):
        print("\t..%d char" % l)
        generator = product(string.digits + string.ascii_lowercase,
                            repeat=int(l))
        p = "**********"
        if p is not False:
            return p

    print('4) Digits + ASCII lower / upper + punctuation')
    # If it fails, we start brute-forcing the 'hard' way
    # Same as possible_char = string.printable[:-5]
    all_char = string.digits + string.ascii_letters + string.punctuation

    for l in range(1, max_nchar + 1):
        print("\t..%d char" % l)
        generator = product(all_char, repeat=int(l))
        p = "**********"
        if p is not False:
            return p


# EXAMPLE
start = time()
bruteforce('sunshine') # Try with '123456' or '751345' or 'test2018'
end = time()
print('Total time: %.2f seconds' % (end - start))