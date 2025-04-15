#date: 2025-04-15T17:14:01Z
#url: https://api.github.com/gists/0de814ae5b97deaf2fbd69bab8c08749
#owner: https://api.github.com/users/rahulbakshee

"""
Given a coke machine with a series of buttons. If you press a button it will get you a certain range of coke. 
Find out if it's possible to get the target range of coke. You can press buttons any number of times.

Example 1:

Input: buttons = [[100, 120], [200, 240], [400, 410]], target = [100, 110]
Output: false
Explanation: if we press first button it might give us 120 volume of coke, not in the target range.
Example 2:

Input: buttons = [[100, 120], [200, 240], [400, 410]], target = [90, 120]
Output: true
Explanation: press first button and you will always get amount of coke in the target range.
Example 3:

Input: buttons = [[100, 120], [200, 240], [400, 410]], target = [300, 360]
Output: true
Explanation: press first and second button and you will always get amount of coke in the target range.
Example 4:

Input: buttons = [[100, 120], [200, 240], [400, 410]], target = [310, 360]
Output: false
Explanation: we can press 1st button 3 times or 1st and 2nd button but it's possible to get only 300, not in the target range.
Example 5:

Input: buttons = [[100, 120], [200, 240], [400, 410]], target = [1, 9999999999]
Output: true
Explanation: you can press any button.


"""

# n = number of buttons, T = upper bound of the target range → target[1]

# time:O(n.T^2) - Each range (s, e) explored is stored in the visited set. In the worst case, 
# you could try every possible pair of start and end values up to T.

# space:O(T^2) - visited set stores O(T²) ranges in the worst case + recursion stack


from typing import List, Tuple
def coke_machine(buttons:[List[List[int]]], target:List[int]):

	def dfs(curr_range:Tuple[int,int])->bool:
		if curr_range in memo:
			return False

		memo.add(curr_range)

		# base case
		if curr_range[0] > target[1]:
			return False	

		if curr_range[0] >= target[0] and curr_range[1] <= target[1]:
			return True

		# otherwise explore the buttons
		for button in buttons:
			start,end = button
			
			new_range = (curr_range[0]+start, curr_range[1]+end)
			if dfs(new_range):
				return True

		return False

	memo = set()
	return dfs((0,0))



buttons = [[100, 120], [200, 240], [400, 410]]
target = [100, 110]
print(coke_machine(buttons, target)) # False


buttons = [[100, 120], [200, 240], [400, 410]]
target = [90, 120]
print(coke_machine(buttons, target)) # True

buttons = [[100, 120], [200, 240], [400, 410]]
target = [300, 360]
print(coke_machine(buttons, target)) # True

buttons = [[100, 120], [200, 240], [400, 410]]
target = [310, 360]
print(coke_machine(buttons, target)) # False

buttons = [[100, 120], [200, 240], [400, 410]]
target = [1, 9999999999]
print(coke_machine(buttons, target)) # True