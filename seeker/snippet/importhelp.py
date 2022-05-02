#date: 2022-05-02T17:09:07Z
#url: https://api.github.com/gists/f506e214453e34797132b9d6cb103d70
#owner: https://api.github.com/users/geniusg94

"""THIS IS addimport.py FILE that will be imported to another file. for method to see below
def add(a,b):
    c=a+b
    return c
def mul(a,b):
    c=a*b
    return c
my_name="GAURAV SHARMA"
"""



'''
there are 3 ways to import variable and functions
#######################method 1
import addimport as f
z=f.my_name
print(z)
functions can be imported the same way
######################method 2
from addimport import *             #*means all thus now use variables directly
z=add(5,6)
print(z)
#####################method 3
from addimport import add,mul,my_name
z=add(5,6)
print(z)
'''



from addimport import add,mul,my_name
z=add(12,13)
print(z)
k=mul(2,3)
print(k)
print(my_name)

"""
OUTPUT OBTAINED
25
6
GAURAV SHARMA
"""
