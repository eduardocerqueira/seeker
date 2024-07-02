#date: 2024-07-02T16:38:58Z
#url: https://api.github.com/gists/666242b3bfe447c6a8757745767a2571
#owner: https://api.github.com/users/gmotzespina

def fibonacci(num):
    if num<= 1:
        return num
    else:
        return fibonacci(num-1) + fibonacci(num-2)

print(fibonacci(6))  # Output: 8