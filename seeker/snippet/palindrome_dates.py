#date: 2021-12-23T16:48:05Z
#url: https://api.github.com/gists/1daf86e346df5f7079baa445d3a3261f
#owner: https://api.github.com/users/izambard

from datetime import datetime

# Palindrome Date - https://www.timeanddate.com/date/palindrome-day.html

#43 21 1234

# Close examples:

#22 02 2022
#12 02 2021
#02 02 2020
#21 02 2012

def gen_digit(min_i: int, max_i :int): yield  from (i for i in range(min_i,max_i+1))

def gen_digit1(): return gen_digit(1,9) # any year, any day
def gen_digit2(): return gen_digit(0,1) # month max is 12 
def gen_digit3(): return gen_digit(0,9)
def gen_digit4(): return gen_digit(0,3) # day max is 31

def gen_year():
    yield from ((i,j,k,l) for i in gen_digit1() for j in gen_digit2() for k in gen_digit3() for l in gen_digit4())

def combine_two(a,b): return int(a*10+b)
def combine_four(a,b,c,d): return int(a*1e3+b*1e2+c*10+d)

def valid_date(i,j,k,l):
    daysInMonth = [[31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]]

    def isleap(i,j,k,l):
        y = combine_four(i,j,k,l)
        return y % 4 == 0 and y % 100 != 0 or y % 400 == 0

    pred1 = not(l == 0 and k == 0) ## No 0 days
    pred2 = not(j >0 and i > 2) # no month higher than 12
    return pred1 and pred2 and not(combine_two(l,k) > daysInMonth[1 if isleap(i,j,k,l) else 0][combine_two(j,i)-1])

def gen_palindromdate():
    for (i,j,k,l) in gen_year():
        if not valid_date(i,j,k,l):
            continue
        else:
            yield (i,j,k,l) 

date_list = []
str_list = []
str_format='%d%m %Y'
g = gen_palindromdate()

try:
    while True:
        (i,j,k,l) = next(g)
        adate = datetime(combine_four(i,j,k,l),combine_two(j,i), combine_two(l,k))
        date_list.append(adate)
        str_list.append(adate.strftime(str_format))
        
except StopIteration as e:
    pass
