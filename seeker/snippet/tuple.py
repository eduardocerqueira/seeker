#date: 2023-04-25T16:52:32Z
#url: https://api.github.com/gists/939761e7f18a24b71492977cc66041a9
#owner: https://api.github.com/users/MipoX

import random
def change(nums):
    index = random.randint(0, 5)
    value = random.randint(100, 1000)
    nums = tuple(list([value if i_len == index else nums[i_len] for i_len in range(len(nums)) ]))
    return nums, value


my_nums = 1, 2, 3, 4, 5

new_nums, rand_val = change(my_nums)
print(new_nums, rand_val)
new_nums = change(new_nums)[0]
rand_val += change(new_nums)[1]

print(new_nums, rand_val)