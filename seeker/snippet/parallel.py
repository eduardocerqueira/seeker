#date: 2021-09-22T17:05:46Z
#url: https://api.github.com/gists/bfab67c2b918c0785ee7e3506f4f7b77
#owner: https://api.github.com/users/neuton

"""
parallel execution function decorator (with quite large overhead)

>>> @parallel(reduction=sum)
... def f(array):
...     return sum(array)
>>> f(range(1, 9))
36

>>> @parallel
... def g(array):
...     return array * 2
>>> g(range(1, 9))
array([ 2,  4,  6,  8, 10, 12, 14, 16])
"""

import functools, os, numpy as np, ray

ray.init(include_dashboard=False, num_cpus=os.cpu_count())


def parallel(n=os.cpu_count(), reduction=np.concatenate):
	def decorator(f):
		rf = ray.remote(f)
		@functools.wraps(f)
		def parallel_f(array, *args, **kwargs):
			# ray.init(include_dashboard=False, num_cpus=n)
			args = [ray.put(arg) for arg in args]
			kwargs = {k: ray.put(v) for k, v in kwargs.items()}
			splitted = np.array_split(array, min(n, len(array)))
			results = ray.get([rf.remote(chunk, *args, **kwargs) for chunk in splitted])
			# ray.shutdown() # lots of overhead
			return reduction(results)
		return parallel_f
	if callable(n):
		return parallel()(n)
	return decorator


def run_tests():
	import warnings
	try:
		import doctest
	except ModuleNotFoundError:
		warnings.warn("No module 'doctest': docstring tests are ignored")
	else:
		assert not doctest.testmod(None if __name__ == '__main__' else __import__(__name__)).failed
	
	@parallel
	def test(array, power=2):
		return array ** power
	test_array = np.random.rand(1000)
	assert np.all(test(test_array, power=3) == test_array ** 3)


if __name__ == '__main__':
	run_tests()