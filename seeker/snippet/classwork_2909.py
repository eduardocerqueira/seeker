#date: 2023-09-29T16:43:48Z
#url: https://api.github.com/gists/14a8689c8625dc82c34c04c1da361f91
#owner: https://api.github.com/users/hirdle

def function_timer(iters):

    def time_decorator(func):
        from time import time as tm

        def wrapper(*args, **kwargs):

            total_time: float = 0
            return_value = 0

            for _ in range(iters):
                start_time: float = tm()
                return_value = func(*args, *kwargs)
                end_time: float = tm()
                total_time += end_time - start_time

            print(f'Время выполнения функции: {total_time/iters:.2} сек')


            return return_value

        return wrapper

    return time_decorator



# 1

@function_timer(100)
def nod_evklid (n, m):

    while n != m:
        if n < m:
            m = m -n
        if n > m:
            n = n -m

    return n


# 2

@function_timer(5)
def sieve_eratosthenes(n):

    list_res = [i for i in range(2, n+1)]

    for p in range(2, n):

        for i in list_res:
            if i % p == 0 and i != p:
                list_res.remove(i)

    return list_res


# 3

@function_timer(5)
def perfect_numbers(num):

    def find_divides(m):
        return sum([i for i in range(1, m) if m % i == 0])

    return [i for i in range(1, num+1) if find_divides(i) == i]


print(perfect_numbers(10000))
print(sieve_eratosthenes(100))
print(nod_evklid(451, 287))