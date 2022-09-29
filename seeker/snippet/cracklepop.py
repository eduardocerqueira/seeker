#date: 2022-09-29T17:32:42Z
#url: https://api.github.com/gists/95a014a472aed135cf64f35d13797f87
#owner: https://api.github.com/users/asbenjamin

"""
        The following orogram prints out the numbers 1 to 100 (inclusive).
        If the number is divisible by 3, print Crackle instead of the number.
        If it's divisible by 5, print Pop.
        If it's divisible by both 3 and 5, print CracklePop.
"""
for number in range(1, 101):
        if number % 15 == 0:
                print ("CracklePop")
                continue
        elif number % 3 == 0:
                print ("Crackle")
                continue
        elif number % 5 == 0:
                print ("Pop")
                continue
        else: print(number)
