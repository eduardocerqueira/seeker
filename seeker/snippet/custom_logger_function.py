#date: 2022-01-21T17:10:19Z
#url: https://api.github.com/gists/905fa117fbf197d03834a872aef1b7eb
#owner: https://api.github.com/users/ssbozy

'''
This is to test function calling and name fetching
'''

import inspect
from datetime import datetime

def slogger(mesg: str) -> None:
	'''
	uses the inspect module to print message with date and caller function
	'''
	d = datetime.now()
	date = d.strftime("%Y-%m-%d %H:%M:%S")
	caller_function = inspect.stack()[1].function
	print(f"{date} - {caller_function} - {mesg}")

def test_func_one() -> None:
	slogger("Hello from test 1")
	return 

def test_func_two() -> None:
	slogger("Hello from test 2")
	test_func_one()
	return 

def main():
	slogger(__name__)
	test_func_one()
	test_func_two()

if __name__ == '__main__':
	main()