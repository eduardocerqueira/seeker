#date: 2023-05-29T17:04:37Z
#url: https://api.github.com/gists/a150cae27d26dd2c4ef6cd7bc4cc7d89
#owner: https://api.github.com/users/seeker10100

import random
from timeit import default_timer as timer
import matplotlib.pyplot as plt
def binary_search(n, a, k, low, high):
 mid = int((low + high) / 2)
 if low > high:
  return -1
 if k == a[mid]:
  return mid
 elif k < a[mid]:
  return binary_search(n, a, k, low, mid - 1)
 else:
  return binary_search(n, a, k, mid + 1, high)
x = []
y = []
for i in range(5):
 # Generate a list of random integers
 n = int(input("\nenter the value of n:"))
 x.append(n)
 arr = [x for x in range(n)]
 k = random.randint(0, n)
 start = timer()
 ind = binary_search(n, arr, k, 0, n - 1)
 end = timer()
 y.append(end - start)
 print("array elements are in the range of 0-",n)
 print("k value=", k)
 print("time taken=", end - start)
 print("element is at the index:", ind)
# Plot the results
plt.plot(x, y)
plt.title('Time Taken for Binary Search')
plt.xlabel('n')
plt.ylabel('Time (seconds)')
plt.show()