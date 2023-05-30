#date: 2023-05-30T16:51:36Z
#url: https://api.github.com/gists/b7a1820a48550f4ee416f8e5b32c0386
#owner: https://api.github.com/users/youngsoul

import random

"""
Monty Hall Simulation


"""
if __name__ == '__main__':
    switch_win_count = 0
    for i in range(0,1000):
        car_door = random.randint(1,3)
        goat_doors = [i for i in [1,2,3] if i != car_door]
        user_door = random.randint(1,3)
        shown_goat_door = random.choice([i for i in goat_doors if i != user_door])
        switched_door = list(set([1,2,3]) - set([shown_goat_door, user_door]))[0]
        if switched_door == car_door:
            switch_win_count += 1
        # print(car_door, goat_doors, user_door, shown_goat_door, switched_door)

    print(f"{switch_win_count}, {(switch_win_count/1000)}")


