#date: 2022-04-27T17:20:03Z
#url: https://api.github.com/gists/85cd4d00d8daff133f72c0fafcd2a4c3
#owner: https://api.github.com/users/PushkraJ99

#Fibonacci Sequence.
def fib(n):
	if (n<=1):
		return n
	else:
		return (fib(n-1)+fib(n-2)) 

num=int(input("Enter number:"))
for u in range(num):
	print(fib(u))