#date: 2023-06-29T16:46:42Z
#url: https://api.github.com/gists/c58e3e39348ee45509e1888921afc5c1
#owner: https://api.github.com/users/BenVosper

from math import inf

from itertools import combinations

# The pool of values you'd like to pick from.
# Can contain duplicate values.
# Does not need to be sorted.
VALUES = [
    2280,
    2250,
    2230,
    2150,
    2130,
    2110,
    2110,
    2040,
    2010,
    2010,
    1989,
    1861,
    1859,
    1835,
    1802,
    1787,
    1766
]

# The size of the groups you'd like.
# We'll make as many groups as we can, discarding the smallest values first.
NUM_MEMBERS = 4

num_groups = len(VALUES) // NUM_MEMBERS
total_num_elements = NUM_MEMBERS * num_groups
sorted_values = sorted(VALUES)
discarded = sorted_values[:-total_num_elements]
values_to_consider = sorted_values[-total_num_elements:]


def group_iterator(s, k):
    """Get unique groups of size 'k' from list 's'.

    Modified from https://cs.stackexchange.com/a/153893
    """
    if len(s) % k != 0:
        raise ValueError("Group does not divide evenly")

    if len(s) // k == 1:
        yield [s]
        return

    remaining_elements = [*s]
    first_item = remaining_elements.pop(0)

    for rest_of_group in combinations(remaining_elements, k - 1):
        rest_of_group = [*rest_of_group]
        group = [first_item, *rest_of_group]

        ungrouped = [*remaining_elements]
        for element in rest_of_group:
            ungrouped.remove(element)

        for other_groups in group_iterator(ungrouped, k):
            yield [group] + other_groups

best_sum_of_absolute_difference = inf
best_groups = None
mean_sum_of_best_groups = None

for groups in group_iterator(values_to_consider, NUM_MEMBERS):
    mean_sum = sum(sum(group) for group in groups) / num_groups
    sum_of_absolute_difference = sum(abs(sum(group) - mean_sum) for group in groups)
    if sum_of_absolute_difference < best_sum_of_absolute_difference:
        best_groups = groups
        best_sum_of_absolute_difference = sum_of_absolute_difference
        mean_sum_of_best_groups = mean_sum


print("Best groups:\n")

for group in best_groups:
    print([*group])
print()

print(f"Mean sum: {mean_sum_of_best_groups}")
print("Difference from mean sum per group:")
print([sum(group) - mean_sum_of_best_groups for group in best_groups])
print(f"Discarded values: {discarded}")




