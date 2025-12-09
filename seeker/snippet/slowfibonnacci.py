#date: 2025-12-09T16:56:27Z
#url: https://api.github.com/gists/ea31be11d71159e57153ee8f0e2b7edf
#owner: https://api.github.com/users/bryankiriama

def slow_fibonacci(n):
    if n == 1 or n == 2:
        return 1
    else:
        return slow_fibonacci(n - 1) + slow_fibonacci(n - 2)

for n in range(1, 31):  # 30 is already quite slow!
    print(n, ":", slow_fibonacci(n))
