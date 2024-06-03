#date: 2024-06-03T17:00:48Z
#url: https://api.github.com/gists/774ba2d7e5f6939016ab844f651ebce0
#owner: https://api.github.com/users/IAnmar0

from Crypto.Util.number import *


def main():
    m1, m2, m3 = 5, 11 ,17
    a1, a2, a3 = 2, 3, 5
    M=m1*m2*m3
    M1=M//m1
    M2=M//m2
    M3=M//m3
    M1_inv= inverse(M1,m1) 
    M2_inv =inverse(M2,m2)
    M3_inv= inverse(M3,m3)

    x=( (a1*M1*M1_inv)+(a2*M2*M2_inv)+(a3*M3*M3_inv) ) %M
    print(x)

if __name__ == "__main__":
    main()