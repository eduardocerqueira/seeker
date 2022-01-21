#date: 2022-01-21T17:14:35Z
#url: https://api.github.com/gists/89a0ec82de3de1147b46037fc508e8a9
#owner: https://api.github.com/users/orel-adivi

import time

def iterative_fibonacci(num):
	current = 1
	prev = 0
	for _ in range(num):
		current = current + prev
		prev = current - prev
	return current

def recursive_fibonacci(num):
	if num == 0 or num == 1:
		return 1
	return recursive_fibonacci(num - 1) + recursive_fibonacci(num - 2)

def memoization_fibonacci(num):
	cache = {}
	
	def auxiliary_fibonacci(num):
		if num in cache:
			return cache[num]
		if num == 0 or num == 1:
			return 1
		value = auxiliary_fibonacci(num - 1) + auxiliary_fibonacci(num - 2)
		cache[num] = value
		return value
	
	return auxiliary_fibonacci(num)

if __name__ == "__main__":
	num = 33
	print("Index of Fibonacci number: ", num)
	
	# iterative implementation:
	start_time = time.time()
	result = iterative_fibonacci(num)
	total_time = time.time() - start_time
	print("Iterative implementation: ", result, " (in ", total_time, " s)")

	# recurcive implementation:
	start_time = time.time()
	result = recursive_fibonacci(num)
	total_time = time.time() - start_time
	print("Recurcive implementation: ", result, " (in ", total_time, " s)")
	
	# recurcive implementation with memoization:
	start_time = time.time()
	result = memoization_fibonacci(num)
	total_time = time.time() - start_time
	print("Recurcive implementation with memoization: ", result, " (in ", total_time, " s)")
