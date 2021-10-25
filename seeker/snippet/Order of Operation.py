#date: 2021-10-25T16:56:41Z
#url: https://api.github.com/gists/d67e2f8fb237d4864580af7d099861a0
#owner: https://api.github.com/users/ROBOMASTER-S1

# Order of Operation-bedmas_list example 1:

# Multiplication and Division always hold
# the presidency over addition and subtraction.
# Use a single asterisk * to create multiplication.

bedmas_list=(
    2*2+4,2+4*2,4+2*2,
    2*2-4,2-4*2,4-2*2,
    2/2+4,2+4/2,4+2/2,
    2/2-4,2-4/2,4-2/2)

for i in bedmas_list:
    print(int(i))

# Order of Operation-bedmas_list example 2:

# exponents always hold the presidency over
# multiplication and division, as well as addition
# and subtraction alike. Use a double asterisk **
# to create exponents

bedmas_list=(
    2*2**4,2**4*2,2/2**4,
    2**4/2,2+2**4,2**4+2,
    2-2**4,2**4-2
    )

for i in bedmas_list:
    print(int(i))