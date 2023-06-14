#date: 2023-06-14T17:04:04Z
#url: https://api.github.com/gists/e876d6830446d4578780881ccb06e7c7
#owner: https://api.github.com/users/aninda052

'''
PROBLEM 1

Given an array nums. We define a running sum of an array as runningSum[i] = sum(nums[0]…nums[i])

Return the running sum of nums.



Example 1:
Input: nums = [1,2,3,4]
Output: [1,3,6,10]

Explanation: Running sum is obtained as follows: [1, 1+2, 1+2+3, 1+2+3+4].

Example 2:
Input: nums = [1,1,1,1,1]
Output: [1,2,3,4,5]
Explanation: Running sum is obtained as follows: [1, 1+1, 1+1+1, 1+1+1+1, 1+1+1+1+1].

Example 3:
Input: nums = [3,1,2,10,1]
Output: [3,4,6,16,17]

'''

def running_sum(input_list = []):

      output_list = []

      prev_inx_value = 0

      for current_value in input_list:
          output_list.append(prev_inx_value + current_value)

          prev_inx_value += current_value

      return output_list
      
      
nums = [1,2,3,4]
print(running_sum(nums))

nums = [1,1,1,1,1]
print(running_sum(nums))

nums = [3,1,2,10,1]
print(running_sum(nums))

