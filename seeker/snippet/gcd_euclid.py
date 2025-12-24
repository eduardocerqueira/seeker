#date: 2025-12-24T17:14:08Z
#url: https://api.github.com/gists/67bf23e5f4f2f0427123671ed3f82478
#owner: https://api.github.com/users/sgouda0412

#! /usr/local/bin/python3.6
"""
G.C.D. Computation with Euclid's algorithm
"""
import sys
import traceback


class GcdEuclid:
    """ Class for G.C.D. Euclid's algorithm """
    
    def gcd(self, a, b):
        """ G.C.D. Calculation

        :param int a: A value
        :param int b: B value
        """
        try:
            return a if b == 0 else self.gcd(b, a % b)
        except Exception as e:
            raise


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("USAGE: ./gcd_euclid.py A B")
        sys.exit(0)
    try:
        a, b = list(map(int, sys.argv[1:3]))
        if a == 0 or b == 0:
            print("Should be integers greater than 0.")
            sys.exit(0)
        obj = GcdEuclid()
        print("GCD({}, {}) = {}".format(a, b, obj.gcd(a, b)))
    except Exception as e:
        traceback.print_exc()
        sys.exit(1)
