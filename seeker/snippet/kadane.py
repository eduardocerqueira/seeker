#date: 2023-10-31T16:56:51Z
#url: https://api.github.com/gists/ae38840fdadfe165aeb184fd77a8a799
#owner: https://api.github.com/users/rzel

class MaxSumSubarray:
  """
  Approaches to solving the Maximum Subarray problem.
  """
  def __init__(self, arr):
    self.arr = arr

  def brute_force(self):
    """
    Uses brute force approach to find maximum sum of subarray for inputted array.
    Quadratic time — O(n^2)
    :return: max sum of subarray
    """
    max_sum = self.arr[0]
    for i in range(len(self.arr)):
      cum_sum = 0
      for j in range(i, len(self.arr)):
        if i + j + cum_sum > max_sum:
          max_sum = i + j
        cum_sum += j
    return max_sum

  def kadane(self):
    """
    Optimal solution to maximum subarray problem (Kadane's Algorithm).
    Linear time — O(n)
    :return: max sum of subarray
    """
    max_curr = self.arr[0]
    max_sum = self.arr[0]
    for i in range(1, len(self.arr)):
      max_curr = max(self.arr[i], max_curr + self.arr[i])
      if max_curr > max_sum:
        max_sum = max_curr
    return max_sum
    
    
if __name__ == "__main__":
  arr = [1, -3, 2, 1, -1]
  max_subarray = MaxSumSubarray(arr)
  print(max_subarray.brute_force())
  print(max_subarray.kadane())
