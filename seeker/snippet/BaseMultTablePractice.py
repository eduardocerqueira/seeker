#date: 2024-02-09T16:55:46Z
#url: https://api.github.com/gists/df6c9e624e156a4741ad7851ae32aab2
#owner: https://api.github.com/users/TannieMielie

import random
import math
import numpy

#GLOBAL CONSTANTS
digits = ['0','1', '2','3', '4','5', '6','7', '8','9', 'a','b', 'c','d', 'e','f', 'g','h', 'i','j']
digitmax = len(digits)

    #FX CONVERT BASE-N VALUE TO DECIMAL
def tbt(base, value):
    finalval = 0
    for i in range(len(value)):
        tempval = base**(len(value)-i-1) * digits.index(value[i])
        finalval += tempval 
    return finalval
    
    #FX CONVERT DECIMAL VALUE TO BASE-N
def bc(base, value):
    if base > digitmax:
        return 'PLEASE INPUT A VALUE LESS THAN OR EQUAL TO ' + str(digitmax)
    #FX VARIABLES
    r = value%base
    q = math.floor(value/base)
    conv = ''
    
    while q != 0:
        rc = digits[r]
        conv = rc + conv
        
        r = q%base
        q = math.floor(q/base)
    rc = digits[r]
    conv = rc + conv
    return conv

b = int(input('Please input base value:'))    
m = str(input('Please input multiplication table value:'))
h = int(input('How many problems do you want to do?'))
score = 0

for i in range(0,h):
    mult = random.randint(1, b)
    prod = tbt(b,m)*mult
    ans = str(input('What is ' + m + ' * ' + str(bc(b,mult)) + '? : '))
    if tbt(b,ans) == prod:
        print('Correct!')
        score += 1
    else:
        print('Incorrect!')

print('Final score: ' + str(score) + '/' + str(h) + " (" + str(round(score/h,3)*100) + '%)')