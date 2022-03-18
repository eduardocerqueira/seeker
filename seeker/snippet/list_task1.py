#date: 2022-03-18T16:48:15Z
#url: https://api.github.com/gists/4a91f06aad719bfb83064acb266ba065
#owner: https://api.github.com/users/kitek83

#list_task1.py
""""Create a function cube_list, that cubes each element of a list.
Call the function with list numbers containing 1 though 10
Show numbers after call """

def cube_list(list1):
    return [num**3 for num in list1] #function cubes each element of a list, returns new list and doesn't change 1st list nums


nums = [x for x in range(1,11)]  # list compprehension - list numbers containing 1 through 10
print(f'nums={nums}')

print(f'cube_list(nums)={cube_list(nums)}')
print(f'nums={nums}')
print()

#2nd method for function cube_list2(), which cubes each element of the list
def cube_list2(values):
    for i in range(len(values)):
        values[i] **= 3             #function change mutable parameter list nums. Function doesn't return new list

#call function and passing list nums
cube_list2(nums)

print(f'nums_cubed_by_cube_list2={nums}')

#let's pass immutable parameter - tuple
immutable = tuple([num for num in range(1,21)]) #use touple method to change mutable parameter (comprehension list) to touple (immutable)
print(f'immutable={immutable}')                 #comprehension list can't be used in touple (comprehension touple)

print(f'cube_list(immutable)={cube_list(immutable)}') #this method takes immutable collection and returns new list
print()

#call function cube_list2(values), which wants to change immutable collection
print(f'immutable={immutable}')
#cube_list2(immutable) --> TypeError: 'tuple' object does not support item assignment













