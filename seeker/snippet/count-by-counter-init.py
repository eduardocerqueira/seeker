#date: 2022-12-06T17:07:02Z
#url: https://api.github.com/gists/4f6015b02aca7ebc9fe1bde9930dc450
#owner: https://api.github.com/users/x0rworld

def count_by_counter_init(random_nums: List[int]) -> Dict:
    counter = Counter(random_nums)
    return counter

random_nums = [random.randrange(1, 10) for _ in range(1000)]
_count_by_counter = count_by_counter_init(random_nums)

# Counter({6: 127, 2: 123, 1: 118, 4: 114, 3: 112, 8: 109, 9: 106, 7: 100, 5: 91})
print(_count_by_counter)