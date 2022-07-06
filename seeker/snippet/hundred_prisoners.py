#date: 2022-07-06T16:52:56Z
#url: https://api.github.com/gists/5bf810d54c0cdcada85b12a1e343535f
#owner: https://api.github.com/users/bedekelly

"""
100 prisoners are each given a number.
A room is arranged with 100 boxes, each containing a unique prisoner's number.
Each prisoner must enter the room, open at most 50 boxes, and leave without
communicating anything. Their aim is to find their own number.
If *every* prisoner finds their own number, they succeed.
If a *single* prisoner fails to find their own number, they fail.

The naive approach of each prisoner opening a random 50 boxes results in an
incredibly low chance of success, (1/2)^100.

An approach exploiting the probability of cycles between box numbers being of
certain lengths increases their chance of success to just under 1/3 with 
communication equally restricted, but each prisoner following a pre-agreed strategy.

This approach is the "looping" function below.
"""
import random

# Prisoners, boxes and slips are each numbered 0 to 99.
boxes = list(range(100))


def looping(number_prisoners=100, number_moves=50):
    """
    Each prisoner first opens the box with their own number.
    They then read the number on the slip in that box, and open
    that box. They continue doing this until they find their own
    number on a slip, or they have exhausted all their tries.
    """
    for prisoner_number in range(number_prisoners):
        found = False
        box_number = prisoner_number
        for _ in range(number_moves):
            if boxes[box_number] == prisoner_number:
                found = True
                break
            box_number = boxes[box_number]
        if not found:
            return False
    return True


def naive(number_prisoners=100, number_moves=50):
    """
    Each prisoner chooses a random assortment of boxes.
    They stop when they have exhausted their tries, or
    when they have found the slip with their number.
    """
    for prisoner_number in range(number_prisoners):
        found = False
        for _ in range(number_moves):
            box_number = random.randint(0, number_prisoners - 1)
            if boxes[box_number] == prisoner_number:
                found = True
                break
        if not found:
            return False
    return True


def experiment(iterations=100_000, run_strategy=naive):
    """
    Run X iterations of an experiment with a given strategy.
    Return the results as a fraction of 1 representing the proportion of successes.
    """
    successes = 0
    for i in range(iterations):
        # if i % 10000 == 0:
        #     print(f"Running trial {i}")
        random.shuffle(boxes)
        successes += 1 if run_strategy() else 0
    return successes / iterations


if __name__ == "__main__":
    print(
        f"Naive strategy gave {experiment(run_strategy=naive) * 100}% chance of success"
    )
    print(
        f"Looping strategy gave {experiment(run_strategy=looping) * 100}% chance of success"
    )
